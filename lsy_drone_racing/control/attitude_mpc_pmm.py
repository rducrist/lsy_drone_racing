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
        self._gates = obs.get("gates_pos")
        self.current_gate_idx = 0
        self._obstacles = obs.get("obstacles_pos")
        self._pos = obs.get("pos")
        self._vel = obs.get("vel")

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        # PMM planner
        self._waypoints = np.vstack((self._pos, self._gates))
        self._start_vel = self._vel
        self._end_vel = np.zeros(3)
        self.traj_t, self.traj_p, self.traj_v, self.traj_a = self.pmm_traj(
            self._waypoints, self._start_vel, self._end_vel, self._dt
        )

        # For visualising using drawline()
        self.traj_pos_viz = self.traj_p
        self.traj_vel_viz = self.traj_v
        self.traj_loaded = True

        self._tick = 0
        self._tick_max = len(self.traj_t) - 1 - self._N
        self._finished = False
        self._config = config

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Computes the control."""
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Setting the initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Build horizon references
        pos_des = self.traj_p[i : i + self._N]
        vel_des = self.traj_v[i : i + self._N]
        acc_des = self.traj_a[i : i + self._N]
        # yaw_des = np.arctan2(vel_des[:, 1], vel_des[:, 0])

        # Fill yref: first nx entries are state; next nu entries are inputs
        yref = np.zeros((self._N, self._ny))
        # State part
        yref[:, 0:3] = pos_des
        # Let roll pitch yaw as zero
        # Vel part
        yref[:, 6:9] = vel_des
        # Let drpy as zero

        # Input part (u reference) goes after state in yref: indices nx ... nx+nu-1
        # Zero roll pitch yaw

        # Set hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        # Warm-start guess for x and u (helps solver converge fast)
        # Build an x_guess consistent with the state order: [pos, rpy, vel, drpy]
        x_guess = np.zeros((self._N, self._nx))
        x_guess[:, 0:3] = pos_des
        x_guess[:, 6:9] = vel_des
        # leave drpy zeros

        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])
            self._acados_ocp_solver.set(j, "x", x_guess[j])

        # Terminal reference (state only)
        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = self.traj_p[i + self._N]
        yref_e[6:9] = self.traj_v[i + self._N]
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
        return False  # continuous control

    def episode_callback(self):
        self._tick = 0
        self._finished = False

    # --------------------- Some helper functions --------------

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
