from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import scipy
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

import os
import csv

from lsy_drone_racing.utils.utils import draw_line

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.ocp_solver import create_acados_model
from lsy_drone_racing.control.ocp_solver import create_ocp_solver
from drone_models.symbols import pos, vel, rpy, drpy

if TYPE_CHECKING:
    from numpy.typing import NDArray

class AttitudeMPC(Controller):
    """Trajectory-generating MPC using attitude control with soft gate/obstacle costs."""
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initializes MPC and pmm planner parameters."""
        super().__init__(obs, info, config)
        self._env_id = config.env.id

        self._N = 30
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Known gate positions
        self.gates = obs.get("gates_pos")
        self.gate_radius = 0.15
        self.current_gate_idx = 0

        # Known obstacles (optional), each as [x,y,z,radius]
        self.obstacles = obs.get("obstacles_pos")

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._finished = False
        self._config = config

        # Cost weight tuning
        self.Q_terminal = np.diag([500, 500, 500, 1, 1, 1, 10, 10, 10, 1, 1, 1])
        self.Q_state = np.diag([500, 500, 500, 1, 1, 1, 10, 10, 10, 1, 1, 1])

        # PMM planner
        self.traj_t = None
        self.traj_pos = None
        self.traj_vel = None
        self.traj_acc = None

        traj_path = "lsy_drone_racing/trajectories/sampled_trajectory.csv"
        self.load_offline_traj_from_csv(traj_path)
        self._time_since_traj_start = 0.0
        
    def load_offline_traj_from_csv(self, csv_path: str) -> None:
        """Load a CSV with columns [t, px, py, pz, vx, vy, vz, ax, ay, az]."""
        t_list, p_list, v_list, a_list = [], [], [], []
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 10:
                    continue
                t = float(row[0])
                px, py, pz = float(row[1]), float(row[2]), float(row[3])
                vx, vy, vz = float(row[4]), float(row[5]), float(row[6])
                ax, ay, az = float(row[7]), float(row[8]), float(row[9])
                t_list.append(t)
                p_list.append([px, py, pz])
                v_list.append([vx, vy, vz])
                a_list.append([ax, ay, az])
        self.traj_t = np.array(t_list)
        self.traj_pos = np.array(p_list)
        self.traj_vel = np.array(v_list)
        self.traj_acc = np.array(a_list)
        if len(self.traj_t) >= 2:
            self._dt_offline = float(self.traj_t[1] - self.traj_t[0])
        else:
            self._dt_offline = None
        self.traj_loaded = True
        self.traj_viz = self.traj_pos[::100]

    def _traj_index_from_time(self, t_now: float) -> int:
        """Return index of trajectory time nearest and not earlier than t_now."""
        if not self.traj_loaded:
            return 0
        # simple nearest index (you can also use bisect/np.searchsorted)
        idx = np.searchsorted(self.traj_t, t_now, side="left")
        idx = max(0, min(idx, len(self.traj_t) - 1))
        return idx

    def _acc_to_rpy_and_thrust(self, acc_des: np.ndarray, yaw_des: float = 0.0):
        """Convert desired acceleration (world frame) to desired roll, pitch, and thrust.

        acc_des: 3-vector desired linear acceleration (m/s^2)
        yaw_des: desired yaw (rad)
        returns (r_des, p_des, y_des, thrust_scalar).
        """
        m = self.drone_params["mass"]
        g_vec = np.array(self.drone_params["gravity_vec"])  # e.g. [0,0,-9.81]
        # Required force in world frame
        F = m * (acc_des - g_vec)  # note signs: a_des - g
        F_norm = np.linalg.norm(F)
        # Avoid division by zero
        if F_norm < 1e-6:
            # hover fallback
            hover_thrust = m * -g_vec[-1]  # consistent with your hover reference
            return 0.0, 0.0, yaw_des, hover_thrust

        b3 = F / F_norm  # desired body z axis in world frame (unit)
        # Decompose to roll (phi) and pitch (theta) for desired yaw psi
        # Standard extraction:
        # phi  = atan2(b3_y, b3_z)
        # theta = atan2(-b3_x, sqrt(b3_y^2 + b3_z^2))
        phi = float(np.arctan2(b3[1], b3[2]))
        theta = float(np.arctan2(-b3[0], np.sqrt(b3[1] ** 2 + b3[2] ** 2)))
        thrust = float(F_norm)

        return phi, theta, yaw_des, thrust

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Computes the control."""
        # Convert quaternion to rpy
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        # Set initial state
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        t_now = self._time_since_traj_start
        self._time_since_traj_start += self._dt

        idx0 = self._traj_index_from_time(t_now)

        # Build horizon references
        for k in range(self._N):
            traj_idx = min(idx0 + k, len(self.traj_t) - 1)
            pos_des = self.traj_pos[traj_idx]
            vel_des = self.traj_vel[traj_idx]
            acc_des = self.traj_acc[traj_idx]
            yaw_des = 0.0  # change if your trajectory contains yaw

            # fill yref: first nx entries are state; next nu entries are inputs
            yref = np.zeros(self._ny)
            # state part
            yref[0:3] = pos_des
            # rpy: compute feedforward rpy from acc
            r_des, p_des, y_des, thrust_ff = self._acc_to_rpy_and_thrust(acc_des, yaw_des)
            yref[3:6] = np.array([r_des, p_des, y_des])
            # vel part
            yref[6:9] = vel_des
            # drpy part left as zeros (or you could approximate from trajectory)
            # input part (u reference) goes after state in yref: indices nx ... nx+nu-1
            yref[self._nx : self._nx + self._nu] = np.array([r_des, p_des, y_des, thrust_ff])
            self._acados_ocp_solver.set(k, "yref", yref)

            # Warm-start guess for x and u (helps solver converge fast)
            # Build an x_guess consistent with the state order: [pos, rpy, vel, drpy]
            x_guess = np.zeros(self._nx)
            x_guess[0:3] = pos_des
            x_guess[3:6] = np.array([r_des, p_des, y_des])
            x_guess[6:9] = vel_des
            # leave drpy zeros or approximate by finite differences if you have it
            self._acados_ocp_solver.set(k, "x", x_guess)
            self._acados_ocp_solver.set(k, "u", np.array([r_des, p_des, y_des, thrust_ff]))

        # Terminal reference (state only)
        traj_idx_end = min(idx0 + self._N, len(self.traj_t) - 1)
        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = self.traj_pos[traj_idx_end]
        yref_e[3:6] = np.array([0.0, 0.0, 0.0])  # fine to push yaw=0 or compute from accel at end
        yref_e[6:9] = np.zeros(3)
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solve MPC
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        predicted_positions = []
        for k in range(self._N + 1):
            x_pred = self._acados_ocp_solver.get(k, "x")
            predicted_positions.append(x_pred[:3])
        self._predicted_traj = np.array(predicted_positions)

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1

        # Update gate if reached
        if np.linalg.norm(obs["pos"] - self.gates[self.current_gate_idx]) < self.gate_radius:
            self.current_gate_idx = min(self.current_gate_idx + 1, len(self.gates) - 1)

        return False  # continuous control

    def episode_callback(self):
        self._tick = 0
        self.current_gate_idx = 0
        self._finished = False