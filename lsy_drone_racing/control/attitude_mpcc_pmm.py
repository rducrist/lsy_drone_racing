"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import time
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from pmm_planner.utils import plan_pmm_trajectory
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mpc_logger import MPCLogger
from lsy_drone_racing.control.mpc_plotter import MPCPlotter
from lsy_drone_racing.control.ocp_solver import create_ocp_solver
from lsy_drone_racing.control.mpcc_solver_config import MPCCSolverConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PmmMPC(Controller):
    """Trajectory-generating MPC using attitude control with soft gate/obstacle costs."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initializes MPC and pmm planner parameters."""
        super().__init__(obs, info, config)
        self._env_id = config.env.id
        mpcc_config = MPCCSolverConfig()

        self._N = 40
        self._T_HORIZON = 0.7
        self._dt = self._T_HORIZON / self._N

        self._update_obs(obs)

        self.drone_params = load_params("so_rpy_rotor", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        # Hover thrust and last_u
        hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        # U = [cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust, dvtheta_cmd]

        # MPCC longitudinal progress states (theta, vtheta)
        self._last_theta = 0.0
        self._last_f_collective = hover_thrust
        self._last_f_cmd = hover_thrust
        self._last_cmd_rpy = np.zeros(3)

        # PMM planner
        self._distance_before = 0.3
        self._distance_after = 0.2
        self._generate_gate_waypoints(
            self._pos, self._current_gate_idx, self._distance_before, self._distance_after
        )
        self._start_vel = self._vel
        self._end_vel = np.array([0.0, 0.0, 0.0])

        self._compute_pmm_traj(self._waypoints, self._start_vel, self._end_vel, self._dt)

        # Precompute arc length along PMM path for MPCC
        diffs = np.diff(self._p_pmm, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)

        s_pmm = np.concatenate(([0.0], np.cumsum(seg_lens)))
        self._delta_theta = mpcc_config.delta_theta

        theta_grid = mpcc_config.theta_grid
        p_of_theta = interp1d(s_pmm, self._p_pmm, axis=0, kind="linear", fill_value="extrapolate")

        pd_list = p_of_theta(theta_grid)
        tp_list = np.zeros_like(pd_list)

        # Like in MPCC paper
        tp_list[1:-1] = (pd_list[2:] - pd_list[:-2]) / (2.0 * self._delta_theta)
        tp_list[0] = (pd_list[1] - pd_list[0]) / self._delta_theta
        tp_list[-1] = (pd_list[-1] - pd_list[-2]) / self._delta_theta

        # Normalize
        tp_norm = np.linalg.norm(tp_list, axis=1, keepdims=True)
        tp_list = tp_list / (tp_norm + 1e-8)

        self._theta_grid = theta_grid
        self._pd_list = pd_list
        self._tp_list = tp_list

        # flattened versions for solver parameters
        self._pd_list_flat = pd_list.reshape(-1)
        self._tp_list_flat = tp_list.reshape(-1)

        # _p_gates = np.hstack((self._gates, self._gates_rpy[:, 2:3]))
        self.p = np.concatenate(
            [
                self._pd_list_flat,
                self._tp_list_flat,
                self._obstacles[:, :2].flatten(),
            ]
        )

        self._initial_position = self._pos.copy()

        # For visualising using drawline()
        self.logger = MPCLogger()
        self.plotter = MPCPlotter(self.logger)

        self.traj_pos_viz = self._p_pmm[::5]
        self.traj_vel_viz = self._v_pmm[::5]

        self._tick = 0
        self._tick_max = len(self._t_pmm) - 1 - self._N

        self._finished = False
        self._config = config

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Computes the control."""
        self._update_obs(obs)

        

        # Set initial state x0 for OCP
        x0 = np.concatenate(
            (
                self._pos,
                self._vel,
                self._rpy,
                np.array([self._last_f_collective, self._last_f_cmd]),
                self._last_cmd_rpy,
                np.array([self._last_theta]),
            )
        )
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Warmstart inputs
        for j in range(self._N + 1):
            self._acados_ocp_solver.set(j, "p", self.p)

        # Solve MPCC
        t_start = time.perf_counter_ns()
        u0, cost = self._solve_mpc()
        t_end = time.perf_counter_ns()

        # Extract next state's theta, vtheta for next iteration
        x_next = self._acados_ocp_solver.get(1, "x")
        self._last_theta = float(x_next[-1])
        self._last_f_collective = float(x_next[9])
        self._last_f_cmd = float(x_next[10])
        self._last_cmd_rpy = x_next[11:14]

        cost = self._acados_ocp_solver.get_cost()

        predictions = self._extract_predictions()

        inner_gate_ring, outer_gate_ring = self._compute_gate_rings(
            self._gates, R.from_quat(self._gates_quat).as_matrix(), 0.1, 0.6, 30
        )
        self.logger.log_step(
            solver_time=(t_end - t_start) * 1e-6,
            cost=cost,
            predictions=predictions,
            state=self._pos,
            control=u0,
            gate_inner_ring=inner_gate_ring,
            gate_outer_ring=outer_gate_ring,
        )
        cmd = np.array(
            [self._last_cmd_rpy[0], self._last_cmd_rpy[1], self._last_cmd_rpy[2], self._last_f_cmd],
            dtype=np.float32,
        )

        return cmd

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """What is being called each sim step."""
        self._tick += 1
        return False  # continuous control

    def episode_callback(self):
        """What has to be called at the end of episode."""
        # self.plotter.plot_solver_times()
        self.plotter.plot_costs()
        self._tick = 0
        self._finished = False
        self._last_theta = 0.0
        self._last_vtheta = 0.0

    # --------------------- Some helper functions --------------
    def _extract_predictions(self) -> NDArray[np.floating]:
        preds = []
        for k in range(self._N + 1):
            x_pred = self._acados_ocp_solver.get(k, "x")
            preds.append(x_pred[:3])
        preds = np.asarray(preds)
        return preds

    def _solve_mpc(self) -> tuple[NDArray[np.floating], np.floating]:
        self._acados_ocp_solver.solve()

        u0 = self._acados_ocp_solver.get(0, "u")
        cost = self._acados_ocp_solver.get_cost()

        return u0, cost

    def _update_obs(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Update internal state from observations."""
        self._gates = obs.get("gates_pos")
        self._gates_quat = obs.get("gates_quat")
        self._gates_rpy = R.from_quat(self._gates_quat).as_euler("xyz")
        self._pos = obs.get("pos")
        self._quat = obs.get("quat")
        self._vel = obs.get("vel")
        self._ang_vel = obs.get("ang_vel")
        self._obstacles = obs.get("obstacles_pos")
        self._current_gate_idx = int(obs.get("target_gate"))
        self._current_obstacle_idx = int(obs.get("obstacles_visited")[-1])
        self._current_gate_pos = self._gates[self._current_gate_idx]
        self._current_gate_quat = self._gates_quat[self._current_gate_idx]
        self._rpy = R.from_quat(self._quat).as_euler("xyz")
        self._drpy = ang_vel2rpy_rates(self._quat, self._ang_vel)

    def _compute_pmm_traj(
        self,
        waypoints: NDArray[np.floating],
        start_vel: NDArray[np.floating],
        end_vel: NDArray[np.floating],
        sampling_period: float,
    ) -> None:
        """Generate a PMM trajectory for a given set of waypoints and start/end velocities."""
        waypoints_config = {
            "start_velocity": start_vel,
            "end_velocity": end_vel,
            "waypoints": waypoints,
        }
        planner_config_file = "./pmm_uav_planner/config/planner/crazyflie.yaml"
        traj = plan_pmm_trajectory(waypoints_config, planner_config_file)

        t_s, p_s, v_s, a_s = traj.get_sampled_trajectory(sampling_period)
        t_s, p_s, v_s, a_s = np.array(t_s), np.array(p_s), np.array(v_s), np.array(a_s)

        self._t_pmm = t_s
        self._p_pmm = p_s
        self._v_pmm = v_s
        self._a_pmm = a_s

        self.traj_pos_viz = self._p_pmm[::5]
        self.traj_vel_viz = self._v_pmm[::5]

    def _project_on_pmm_path(self, pos: NDArray[np.floating]) -> tuple[float, int]:
        """Project current position onto PMM path (in a nearest-neighbor sense) and return (s_cur, index)."""
        dists = np.linalg.norm(self._p_pmm - pos[None, :], axis=1)
        idx = int(np.argmin(dists))
        s_cur = self._s_pmm[idx]
        return s_cur, idx

    def _generate_gate_waypoints(
        self,
        start_pos: NDArray[np.floating],
        start_gate_idx: int,
        distance_before: float,
        distance_after: float,
    ) -> None:
        """This function generates a set of waypoints for each gate starting from current gate index."""
        waypoints = [start_pos.copy()]  # start at drone

        # validate start_gate_idx
        n_gates = len(self._gates)
        for i in range(start_gate_idx, n_gates):
            gate_pos = self._gates[i]
            gate_quat = self._gates_quat[i]
            R_gate = R.from_quat(gate_quat).as_matrix()
            gate_forward = R_gate[:, 0]  # x-axis of gate frame

            wp_before = gate_pos - distance_before * gate_forward
            wp_after = gate_pos + distance_after * gate_forward

            waypoints.append(wp_before)
            waypoints.append(gate_pos)
            waypoints.append(wp_after)
            if i ==2:
                wp_safe = wp_after + np.array([0.0,0.0,0.5])
                waypoints.append(wp_safe)

        self._waypoints = np.vstack(waypoints)

    def _replan_trajectory(self) -> None:
        """Re-generate PMM trajectory when gates move."""
        self._generate_gate_waypoints(
            self._pos, self._current_gate_idx, self._distance_before, self._distance_after
        )
        self._start_vel = self._vel
        self._compute_pmm_traj(self._waypoints, self._start_vel, self._end_vel, self._dt)

        # Update visualization
        self.traj_pos_viz = self._p_pmm[::5]
        self.traj_vel_viz = self._v_pmm[::5]

        # Reset tick / horizon index
        self._tick = 0
        self._tick_max = max(0, len(self._t_pmm) - 1 - self._N)

        # Remember last gate position
        self._last_gate_pos = self._current_gate_pos.copy()

    def _compute_gate_rings(
        self, gate_c, R, r_i, r_o, n_pts=60
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute inner and outer annulus rings of a gate in world frame.

        Args:
            gate_c (array-like, shape (3,)):
                Gate center in world coordinates
            R (array-like, shape (3,3)):
                Rotation matrix from gate frame -> world frame
                (columns are gate-frame axes in world frame)
            r_i (float):
                Inner radius
            r_o (float):
                Outer radius
            n_pts (int):
                Number of points per ring
        """
        beta = np.linspace(0.0, 2.0 * np.pi, n_pts)

        # gate-frame points (X axis is normal to gate plane)
        ring_i_gate = np.stack(
            [np.zeros_like(beta), r_i * np.cos(beta), r_i * np.sin(beta)], axis=1
        )

        ring_o_gate = np.stack(
            [np.zeros_like(beta), r_o * np.cos(beta), r_o * np.sin(beta)], axis=1
        )

        ring_inner = np.zeros((4, n_pts, 3))
        ring_outer = np.zeros((4, n_pts, 3))
        # transform to world frame
        for i in range(4):
            ring_inner[i] = (R[i] @ ring_i_gate.T).T + gate_c[i]
            ring_outer[i] = (R[i] @ ring_o_gate.T).T + gate_c[i]

        return ring_inner, ring_outer
