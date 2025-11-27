"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from pmm_planner.utils import plan_pmm_trajectory
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.ocp_solver import create_ocp_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray



class AttitudeMPC(Controller):
    """Trajectory-generating MPC using attitude control with soft gate/obstacle costs."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initializes MPC and pmm planner parameters."""
        super().__init__(obs, info, config)
        self._env_id = config.env.id

        self._N = 20
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
        self.traj_p = None
        self.traj_v = None
        self.traj_a = None

        # For visualising using drawline()
        self.traj_viz = None
        self.traj_loaded = False

        self._time_since_traj_start = 0.0

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

        waypoints = np.vstack((obs.get("pos"), obs.get("gates_pos")))
        start_vel = obs.get("vel")
        end_vel = np.array([0.0, 0.0, 0.0])
        self.traj_t, self.traj_p, self.traj_v, self.traj_a = self.pmm_traj(
            waypoints, start_vel, end_vel, self._dt
        )

        idx0 = self._traj_index_from_time(self.traj_t, t_now)

        # Build horizon references
        for k in range(self._N):
            pos_des = self.traj_p[k]
            vel_des = self.traj_v[k]
            acc_des = self.traj_a[k]

            yaw_des = np.arctan2(vel_des[1], vel_des[0])

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
        yref_e[0:3] = self.traj_p[traj_idx_end]
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

    def _traj_index_from_time(self, t_s: float, t_now: float) -> int:
        """Return index of trajectory time nearest and not earlier than t_now."""
        if not self.traj_loaded:
            return 0
        # simple nearest index (you can also use bisect/np.searchsorted)
        idx = np.searchsorted(t_s, t_now, side="left")
        idx = max(0, min(idx, len(t_s) - 1))
        return idx

    # --------------------- Some helper functions --------------
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

    def pmm_traj(self, waypoints, start_vel, end_vel, sampling_period):
        """Generate a pmm trajectory for a given set of waypoints and start and end velocities."""
        waypoints_config = {
            "start_velocity": start_vel,
            "end_velocity": end_vel,
            "waypoints": waypoints,
        }

        planner_config_file = "./pmm_uav_planner/config/planner/crazyflie.yaml"
        traj = plan_pmm_trajectory(waypoints_config, planner_config_file)

        t_s, p_s, v_s, a_s = traj.get_sampled_trajectory(sampling_period)
        t_s, p_s, v_s, a_s = np.array(t_s), np.array(p_s), np.array(v_s), np.array(a_s)
        
        self.traj_viz = p_s
        self.traj_viz[::100]

        self.traj_loaded = True

        return t_s, p_s, v_s, a_s
