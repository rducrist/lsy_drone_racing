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
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mpc_logger import MPCLogger
from lsy_drone_racing.control.mpc_plotter import MPCPlotter
from lsy_drone_racing.control.ocp_solver import create_ocp_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PmmMPC(Controller):
    """Trajectory-generating MPC using attitude control with soft gate/obstacle costs."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initializes MPC and pmm planner parameters."""
        super().__init__(obs, info, config)
        self._env_id = config.env.id

        self._N = 20
        self._dt = 1 / config.env.freq
        # self._T_HORIZON = self._N * self._dt
        self._T_HORIZON = 0.7

        self._update_obs(obs)
        self._last_gate_pos = self._gates[self._current_gate_idx].copy()

        self.corridor = np.array([0.05, 0.05, 0.05])
        self.corridor_default = np.array([10.0, 10.0, 10.0])
        self.gate_influence_radius = 1
        self._sensor_range = 0.65

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
    

        # Hover thrust and last_u
        hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        # U = [cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust, dvtheta_cmd]
        self.last_u = np.array([0.0, 0.0, 0.0, hover_thrust, 0.0])


        # MPCC longitudinal progress states (theta, vtheta)
        self._last_theta = 0.0
        self._last_vtheta = 0.0

        # PMM planner
        self._distance_before = 0.3
        self._distance_after = 0.3
        self._generate_gate_waypoints(
            self._pos, self._current_gate_idx, self._distance_before, self._distance_after
        )
        self._start_vel = self._vel
        self._end_vel = np.array([0.0, 0.0, 0.0])

        self._compute_pmm_traj(self._waypoints, self._start_vel, self._end_vel, self._dt)

        # For visualising using drawline()
        self.logger = MPCLogger()
        self.plotter = MPCPlotter(self.logger)

        self.traj_pos_viz = self._p_pmm[::5]
        self.traj_vel_viz = self._v_pmm[::5]

        self._tick = 0
        self._tick_max = len(self._t_pmm) - 1 - self._N

        self._finished = False
        self._config = config

        # MPCC weights (used in parameter p[j, 6:9])
        self._qc = 10.0
        self._ql = 1.0
        self._mu = 1.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Computes the control."""
        self._update_obs(obs)
    
        # Initial progress state based on closest point on PMM path
        s_cur, _ = self._project_on_pmm_path(self._pos)

        if self._tick == 0:
            theta0 = s_cur
            # simple initial guess for vtheta: projection of vel onto path tangent
            vtheta0 = max(0.1, np.linalg.norm(self._vel))
        else:
            theta0 = self._last_theta
            vtheta0 = self._last_vtheta

        # Set initial state x0 for OCP
        x0 = np.concatenate((self._pos, self._rpy, self._vel, self._drpy, np.array([theta0, vtheta0])))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # # compute current state
        # i = min(self._tick, self._tick_max)
        # if self._tick >= self._tick_max:
        #     self._finished = True

        dist_to_gate = np.linalg.norm(self._current_gate_pos - self._pos)
        gate_moved = np.linalg.norm(self._last_gate_pos - self._current_gate_pos) > 0.001
        entered_sensor_range = dist_to_gate < self._sensor_range

        if gate_moved and entered_sensor_range:
            self._replan_trajectory()


        # Build MPCC parameter horizon & warm-start states
        p_horizon = self._build_mpcc_parameters(theta0, vtheta0)
        x_guess = self._build_x_guess(theta0, vtheta0)

        # Apply parameters and warm-starts
        self._apply_constraints(p_horizon, x_guess)


        # Solve MPC
        t_start = time.perf_counter_ns()
        u0, cost = self._solve_mpc()
        t_end = time.perf_counter_ns()

        # Extract next state's theta, vtheta for next iteration
        x1 = self._acados_ocp_solver.get(1, "x")
        self._last_theta = float(x1[-2])
        self._last_vtheta = float(x1[-1])

        predictions = self._extract_predictions()

        self.logger.log_step(
            solver_time=(t_end - t_start) * 1e-6,
            cost=cost,
            predictions=predictions,
            state=self._pos,
            control=u0
        )
        self.last_u = u0.copy()
        print(f"theta0={theta0:.3f}, vtheta0={vtheta0:.3f}, u0={u0}")
        return u0[:4]

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

    # MPCC-specific functions
    def _build_mpcc_parameters(self, theta0: float, vtheta0: float) -> NDArray[np.floating]:
        """
        Build per-stage MPCC parameter vectors

            p_j = [p_ref(3), t_ref(3), qc, ql, mu, obs_1(2), obs_2(2), obs_3(2), obs_4(2)]
                 ∈ ℝ¹⁷

        using theta0, vtheta0 and the PMM path and the current obstacle positions.
        """            
        p_horizon = np.zeros((self._N, 17))

        for j in range(self._N):
            s_j = theta0 + j * vtheta0 * self._dt
            p_ref, t_ref = self._pos_and_tangent_from_s(s_j)

            p_horizon[j, 0:3] = p_ref          # p_ref
            p_horizon[j, 3:6] = t_ref          # t_ref
            p_horizon[j, 6]   = self._qc       # qc
            p_horizon[j, 7]   = self._ql       # ql
            p_horizon[j, 8]   = self._mu       # mu

        # --- Obstacle-Teil: 8 Einträge, für alle Stages gleich ---
        # self._obstacles: shape (4, 3) → wir nehmen nur (x, y)
        obs_xy = np.zeros(8)
        num_obs = min(4, len(self._obstacles))
        for k in range(num_obs):
            obs_xy[2 * k : 2 * k + 2] = self._obstacles[k][:2]

        # dieselben Obstacle-Parameter für alle Stages
        p_horizon[:, 9:17] = obs_xy[None, :]

        return p_horizon
    
    def _build_x_guess(self, theta0: float, vtheta0: float) -> NDArray[np.floating]:
        """
        Build a simple warm-start for the state trajectory:
            pos, vel from PMM path,
            rpy, drpy zero,
            theta, vtheta linearly increasing.
        """
        x_guess = np.zeros((self._N, self._nx))

        for j in range(self._N):
            s_j = theta0 + j * vtheta0 * self._dt
            p_ref, _ = self._pos_and_tangent_from_s(s_j)

            x_guess[j, 0:3] = p_ref                      # pos
            # rpy (3:6) left at zero
            # vel (6:9) - approximate via finite differences along PMM path:
            if j < self._N - 1:
                s_next = theta0 + (j + 1) * vtheta0 * self._dt
                p_next, _ = self._pos_and_tangent_from_s(s_next)
                v_approx = (p_next - p_ref) / self._dt
                x_guess[j, 6:9] = v_approx
            # drpy (9:12) left at zero
            x_guess[j, 12] = s_j                         # theta
            x_guess[j, 13] = vtheta0                     # vtheta

        return x_guess

    def _apply_constraints(
        self,
        p_horizon: NDArray[np.floating],
        x_guess: NDArray[np.floating],
    ) -> None:
        """
        Apply MPCC parameters and warm-start x/u into the acados solver.
        """
        for j in range(self._N):
            stage = j + 1

            # set MPCC parameters for stage j
            self._acados_ocp_solver.set(stage, "p", p_horizon[j])

            # warm-start state
            self._acados_ocp_solver.set(stage, "x", x_guess[j])

            # IMPORTANT: do not set u at terminal stage
            if stage < self._N:
                self._acados_ocp_solver.set(stage, "u", self.last_u)

    def _update_obs(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Update internal state from observations."""
        self._gates = obs.get("gates_pos")
        self._gates_quat = obs.get("gates_quat")
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

        # Precompute arc length along PMM path for MPCC
        diffs = np.diff(self._p_pmm, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self._s_pmm = np.concatenate(([0.0], np.cumsum(seg_lens)))
        self._s_total = float(self._s_pmm[-1])
        self.traj_pos_viz = self._p_pmm[::5]
        self.traj_vel_viz = self._v_pmm[::5]

    def _project_on_pmm_path(self, pos: NDArray[np.floating]) -> tuple[float, int]:
        """ Project current position onto PMM path (in a nearest-neighbor sense) and return (s_cur, index). """
        dists = np.linalg.norm(self._p_pmm - pos[None, :], axis=1)
        idx = int(np.argmin(dists))
        s_cur = self._s_pmm[idx]
        return s_cur, idx

    def _pos_and_tangent_from_s(self, s: float) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """ Get interpolated position and tangent on PMM path for a given arc length s. Linear interpolation between two neighboring PMM samples. """
        # clamp s into valid range
        s_clamped = np.clip(s, 0.0, max(self._s_total - 1e-6, 0.0))

        # find segment index
        idx = int(np.searchsorted(self._s_pmm, s_clamped, side="right") - 1)
        idx = np.clip(idx, 0, len(self._s_pmm) - 2)
        s0 = self._s_pmm[idx]
        s1 = self._s_pmm[idx + 1]
        p0 = self._p_pmm[idx]
        p1 = self._p_pmm[idx + 1]
        if s1 - s0 > 1e-9:
            tau = (s_clamped - s0) / (s1 - s0)
        else:
            tau = 0.0
        p_ref = (1.0 - tau) * p0 + tau * p1
        t_vec = p1 - p0
        norm_t = np.linalg.norm(t_vec)
        if norm_t < 1e-9:
            t_ref = np.array([1.0, 0.0, 0.0])
        else:
            t_ref = t_vec / norm_t
        return p_ref, t_ref

    def _generate_gate_waypoints(
        self,
        start_pos: NDArray[np.floating],
        start_gate_idx: int,
        distance_before: float,
        distance_after: float,
    ) -> None:
        """Generate a set of waypoints for each gate starting from the current gate index."""
        waypoints = [start_pos.copy()]  # start at drone
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
        self._waypoints = np.vstack(waypoints)

    def _replan_trajectory(self) -> None:
        """Re-generate PMM trajectory when gates move."""
        self._generate_gate_waypoints(
            self._pos,
            self._current_gate_idx,
            self._distance_before,
            self._distance_after
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
