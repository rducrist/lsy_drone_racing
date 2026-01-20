"""This module implements a Model Predictive Contouring Control (MPCC) framework for quadrotor flight using attitude-level control.

A Point Mass Model(PMM) planner generates a time optimal
reference trajectory through gate centers, which is reparameterized and tracked
by an MPCC. The controller supports online replanning, soft gate and obstacle costs, and is designed for drone racing
scenarios with moving gates.
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
from lsy_drone_racing.control.mpcc_solver_config import MPCCSolverConfig
from lsy_drone_racing.control.ocp_solver import create_ocp_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPCC_PMM(Controller):
    """Trajectory-generating MPCC using attitude control with soft gate/obstacle costs."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initializes MPCC and PMM planner parameters."""
        super().__init__(obs, info, config)
        self._env_id = config.env.id
        self._mpcc_config = MPCCSolverConfig()

        self._N = self._mpcc_config.N
        self._T_HORIZON = self._mpcc_config.T_horizon
        self._dt = self._mpcc_config.dt

        # Get first observations
        self._update_obs(obs)
        self._initial_position = self._pos.copy()

        self.drone_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        # Define hover thrust
        hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        # Initialize MPCC progress variables
        self._last_theta = 0.0
        self._last_f_collective = hover_thrust
        self._last_f_cmd = hover_thrust
        self._last_cmd_rpy = np.zeros(3)

        # PMM planner
        self._distance_before = self._mpcc_config.distance_before
        self._distance_after = self._mpcc_config.distance_after
        self._generate_gate_waypoints(self._distance_before, self._distance_after)
        self._start_vel = self._vel
        self._end_vel = self._mpcc_config.end_vel

        self._compute_pmm_traj(self._waypoints, self._start_vel, self._end_vel, self._dt)

        self._parametrize_trajectory(
            self._p_pmm, self._mpcc_config.theta_grid, self._mpcc_config.delta_theta
        )
        self._qc_dyn_from_gates()
        # Replanning params
        self._last_gate_pos = self._current_gate_pos
        self._last_gate_idx = self._current_gate_idx
        self._sensor_range = self._mpcc_config.sensor_range

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

        # Replan only if gate position actually updated
        gate_switched = self._current_gate_idx != self._last_gate_idx
        gate_moved = np.linalg.norm(self._current_gate_pos - self._last_gate_pos) > 0.01
        committed = np.linalg.norm(self._current_gate_pos - self._pos) < self._sensor_range

        if gate_moved and not gate_switched and not committed:
            self._replan_trajectory()

            theta_proj, _ = self._find_closest_point_linear(self._p_pmm, self._s_pmm, self._pos)
            self._last_theta = max(self._last_theta, float(theta_proj))

        self._last_gate_idx = self._current_gate_idx
        self._last_gate_pos = self._current_gate_pos.copy()

        # Set initial state x0 for OCP
        x0 = np.concatenate(
            (
                self._pos,
                self._vel,
                self._rpy,
                np.array([self._last_f_collective]),
                self._last_cmd_rpy,
                np.array([self._last_f_cmd]),
                np.array([self._last_theta]),
            )
        )

        if not hasattr(self, "_x_warm"):
            self._x_warm = [x0.copy() for _ in range(self._N + 1)]
            self._u_warm = [np.zeros(self._nu) for _ in range(self._N)]
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1]]
            self._u_warm = self._u_warm[1:] + [self._u_warm[-1]]
        
        for i in range(self._N):
            self._acados_ocp_solver.set(i, "x", self._x_warm[i])
            self._acados_ocp_solver.set(i, "u", self._u_warm[i])
        self._acados_ocp_solver.set(self._N, "x", self._x_warm[self._N])

        # Set initial guess
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Set parameter vector
        self._p = np.concatenate(
            [self._pd_list, self._tp_list, self._qc_dyn, self._obstacles[:, :2].flatten()]
        )

        for j in range(self._N + 1):
            self._acados_ocp_solver.set(j, "p", self._p)

        # Solve MPCC
        t_start = time.perf_counter_ns()
        u0, cost = self._solve_mpc()
        t_end = time.perf_counter_ns()

        # Update warmstart
        self._x_warm = [self._acados_ocp_solver.get(i, "x") for i in range(self._N + 1)]
        self._u_warm = [self._acados_ocp_solver.get(i, "u") for i in range(self._N)]

        # Extract next state's theta, vtheta for next iteration
        x_next = self._acados_ocp_solver.get(1, "x")
        self._last_theta = float(x_next[-1])
        self._last_f_collective = float(x_next[9])
        self._last_f_cmd = float(x_next[13])
        self._last_cmd_rpy = x_next[10:13]

        # For visualisation
        predictions = self._extract_predictions()

        self.logger.log_step(
            solver_time=(t_end - t_start) * 1e-6,
            cost=cost,
            predictions=predictions,
            state=self._pos,
            control=u0,
        )

        # Build command vector
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
        # self.plotter.plot_costs()
        self._tick = 0
        self._finished = False
        self._last_theta = 0.0
        self._last_vtheta = 0.0
        self._acados_ocp_solver.reset()


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

    def _generate_gate_waypoints(self, distance_before: float, distance_after: float) -> None:
        """This function generates a set of waypoints for each gate starting from current gate index."""
        waypoints = [self._initial_position]  # start at drone
        take_off_wp = waypoints[0] + np.array([0.0,0.0,0.1])
        waypoints.append(take_off_wp)

        # validate start_gate_idx
        n_gates = len(self._gates)
        for i in range(n_gates):
            gate_pos = self._gates[i]
            gate_quat = self._gates_quat[i]
            R_gate = R.from_quat(gate_quat).as_matrix()
            gate_forward = R_gate[:, 0]  # x-axis of gate frame

            wp_before = gate_pos - distance_before * gate_forward
            wp_after = gate_pos + distance_after * gate_forward

            waypoints.append(wp_before)
            waypoints.append(gate_pos)
            waypoints.append(wp_after)
            if i == 2:
                wp_extra = wp_before + np.array([0.0,-0.1,0.3]) # small hack make pmm trajectory feasible
                waypoints.append(wp_extra)

        self._waypoints = np.vstack(waypoints)

    def _replan_trajectory(self) -> None:
        """Re-generate PMM trajectory when gates move."""
        self._generate_gate_waypoints(self._distance_before, self._distance_after)
        self._start_vel = np.zeros(3)
        self._compute_pmm_traj(self._waypoints, self._start_vel, self._end_vel, self._dt)

        self._parametrize_trajectory(
            self._p_pmm, self._mpcc_config.theta_grid, self._mpcc_config.delta_theta
        )

        self._qc_dyn_from_gates()

        # Update visualization
        self.traj_pos_viz = self._p_pmm[::5]
        self.traj_vel_viz = self._v_pmm[::5]

        # Remember last gate position
        self._last_gate_pos = self._current_gate_pos.copy()

    def _find_closest_point_linear(
        self,
        p_points: np.ndarray,  # (N, 3)
        s_points: np.ndarray,  # (N,)
        position: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Exact projection of a point onto a piecewise-linear, arc-length parameterized path."""
        best_dist2 = np.inf
        best_theta = 0.0
        best_point = p_points[0]

        for i in range(len(p_points) - 1):
            p0 = p_points[i]
            p1 = p_points[i + 1]
            d = p1 - p0
            seg_len2 = np.dot(d, d)

            if seg_len2 < 1e-12:
                continue

            alpha = np.dot(position - p0, d) / seg_len2
            alpha = np.clip(alpha, 0.0, 1.0)

            p_proj = p0 + alpha * d
            dist2 = np.dot(position - p_proj, position - p_proj)

            if dist2 < best_dist2:
                best_dist2 = dist2
                best_point = p_proj
                best_theta = s_points[i] + alpha * np.sqrt(seg_len2)

        return best_theta, best_point

    def _parametrize_trajectory(self, pmm_path: NDArray, theta_grid: NDArray, delta_theta: float):
        # Precompute arc length along PMM path for MPCC
        diffs = np.diff(pmm_path, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)

        self._s_pmm = np.concatenate(([0.0], np.cumsum(seg_lens)))

        p_of_theta = interp1d(
            self._s_pmm, pmm_path, axis=0, kind="linear", fill_value="extrapolate"
        )

        pd_list = p_of_theta(theta_grid)
        tp_list = np.zeros_like(pd_list)

        # Like in MPCC paper
        tp_list[1:-1] = (pd_list[2:] - pd_list[:-2]) / (2.0 * delta_theta)
        tp_list[0] = (pd_list[1] - pd_list[0]) / delta_theta
        tp_list[-1] = (pd_list[-1] - pd_list[-2]) / delta_theta

        # Normalize
        tp_norm = np.linalg.norm(tp_list, axis=1, keepdims=True)
        tp_list = tp_list / (tp_norm + 1e-8)

        theta_grid = theta_grid
        pd_list = pd_list
        tp_list = tp_list

        # flattened versions for solver parameters
        self._pd_list = pd_list.reshape(-1)
        self._tp_list = tp_list.reshape(-1)


    def _qc_dyn_from_gates(self) -> NDArray:
        # pd_list: (M,3)
        M = self._mpcc_config.M
        pdM = self._pd_list.reshape(M, 3)
        qc = np.zeros(pdM.shape[0], dtype=float)
        for g in self._gates:
            d = np.linalg.norm(pdM - g, axis=1)
            qc = np.maximum(qc, np.exp(-5.0 * d * d))
        self._qc_dyn = qc

        
