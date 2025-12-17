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
        self._last_gate_pos = self._current_gate_pos
        self._last_gate_idx = self._current_gate_idx

        self.corridor = np.array([0.05, 0.05, 0.05])
        self.corridor_default = np.array([10.0, 10.0, 10.0])
        self.gate_influence_radius = 1
        self._sensor_range = 0.5

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        self.last_u = np.array(
            [self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1], 0.0, 0.0, 0.0]
        )

        # PMM planner
        self._distance_before = 0.2
        self._distance_after = 0.2
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

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Computes the control."""
        self._update_obs(obs)

        # Setting the initial state
        x0 = np.concatenate((self._pos, self._rpy, self._vel, self._drpy))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)


        
        gate_switched = self._current_gate_idx != self._last_gate_idx
        gate_moved    = np.linalg.norm(self._current_gate_pos - self._last_gate_pos) > 0.01
        committed     = np.linalg.norm(self._current_gate_pos - self._pos) < self._sensor_range


        if gate_moved and not gate_switched and not committed:
            self._replan_trajectory()

        self._last_gate_idx = self._current_gate_idx
        self._last_gate_pos = self._current_gate_pos.copy()

         # compute current state
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Build horizon references
        yref, yref_e, x_guess = self._build_references(i)

    
        # Apply corridor constraints and warm-starts
        self._apply_constraints(i, yref, yref_e, x_guess)

        # Solve MPC
        t_start = time.perf_counter_ns()
        u0, cost = self._solve_mpc()
        t_end = time.perf_counter_ns()

        predictions = self._extract_predictions()

        inner_gate_ring, outer_gate_ring = self._compute_gate_rings(self._gates, R.from_quat(self._gates_quat).as_matrix(), 0.1, 0.6, 30)

        self.logger.log_step(
            solver_time=(t_end - t_start) * 1e-6,
            cost=cost,
            predictions=predictions,
            state=self._pos,
            control=u0,
            gate_inner_ring = inner_gate_ring,
            gate_outer_ring=outer_gate_ring
        )
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
        """What is being called each sim step."""
        self._tick += 1
        return False  # continuous control

    def episode_callback(self):
        """What has to be called at the end of episode."""
        # self.plotter.plot_solver_times()
        self.plotter.plot_costs()
        self._tick = 0
        self._finished = False

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

    def _apply_constraints(
        self,
        i: int,
        yref: NDArray[np.floating],
        yref_e: NDArray[np.floating],
        x_guess: NDArray[np.floating],
    ) -> None:
        """Apply corridor constraints and initialize yref / x_guess in the solver."""
        dist_to_gate = np.linalg.norm(self._current_gate_pos - self._pos)

        R_gate = R.from_quat(self._current_gate_quat).as_matrix()
        corridor_rot = R_gate @ self.corridor

        if dist_to_gate < self.gate_influence_radius:
            corridor = corridor_rot
        else:
            corridor = self.corridor_default

        for j in range(self._N):
            stage = j + 1

            # lbx_j = self._ocp.constraints.lbx.copy()
            # ubx_j = self._ocp.constraints.ubx.copy()

            # lbx_j[:3] = self._current_gate_pos - corridor
            # ubx_j[:3] = self._current_gate_pos + corridor

            # if stage <= self._N - 1:
            # self._acados_ocp_solver.set(stage, "lbx", lbx_j)
            # self._acados_ocp_solver.set(stage, "ubx", ubx_j)

            n_gates = self._gates.shape[0]
            n_obs = self._obstacles.shape[0]

            p = np.zeros(4*n_gates + 2*n_obs)

            # gates
            for g in range(n_gates):
                p[4*g:4*g+4] = np.hstack([self._gates[g], self._gates_rpy[g][2]])

            offset = 4*n_gates

            # obstacles
            for o in range(n_obs):
                p[offset + 2*o : offset + 2*o + 2] = self._obstacles[o][2:]

            self._acados_ocp_solver.set(stage, "p", p)

            self._acados_ocp_solver.set(j, "yref", yref[j])
            self._acados_ocp_solver.set(j, "x", x_guess[j])
            self._acados_ocp_solver.set(j, "u", self.last_u)

        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

    def _build_references(
        self, i: int
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Build yref, terminal yref_e, and warm-start x_guess."""
        pos_des = self._p_pmm[i : i + self._N]
        
        vel_des = self._v_pmm[i : i + self._N]

        # Stage references
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = pos_des
        yref[:, 6:9] = vel_des
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        # Terminal reference
        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = self._p_pmm[i + self._N]
        yref_e[6:9] = self._v_pmm[i + self._N]

        # Warm-start guess
        x_guess = np.zeros((self._N, self._nx))
        x_guess[:, 0:3] = pos_des
        x_guess[:, 6:9] = vel_des

        return yref, yref_e, x_guess

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

        self._t_pmm = t_s
        self._p_pmm = p_s
        self._v_pmm = v_s
        self._a_pmm = a_s

        self.traj_pos_viz = self._p_pmm[::5]
        self.traj_vel_viz = self._v_pmm[::5]

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

        self._waypoints = np.vstack(waypoints)

    def _replan_trajectory(self) -> None:
        """Re-generates a pmm trajectory."""
        self._generate_gate_waypoints(
            self._pos, self._current_gate_idx, self._distance_before, self._distance_after
        )
        self._start_vel = self._vel.copy()
        self._compute_pmm_traj(self._waypoints, self._start_vel, self._end_vel, self._dt)

        # Update visualization
        self.traj_pos_viz = self._p_pmm[::5]
        self.traj_vel_viz = self._v_pmm[::5]

        # Reset tick
        self._tick = 0
        self._tick_max = max(0, len(self._t_pmm) - 1 - self._N)

        # Remember last gate position
        self._last_current_gate_pos = self._current_gate_pos.copy()

    def _compute_gate_rings(self, gate_c, R, r_i, r_o, n_pts=60) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
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
        theta = np.linspace(0.0, 2.0 * np.pi, n_pts)

        # gate-frame points (X axis is normal to gate plane)
        ring_i_gate = np.stack(
            [np.zeros_like(theta), r_i * np.cos(theta), r_i * np.sin(theta)], axis=1
        )

        ring_o_gate = np.stack(
            [np.zeros_like(theta), r_o * np.cos(theta), r_o * np.sin(theta)], axis=1
        )

        ring_inner = np.zeros((4, n_pts, 3))
        ring_outer = np.zeros((4, n_pts, 3))
        # transform to world frame
        for i in range(4):
            ring_inner[i] = (R[i] @ ring_i_gate.T).T + gate_c[i]
            ring_outer[i] = (R[i] @ ring_o_gate.T).T + gate_c[i]

        return ring_inner, ring_outer
