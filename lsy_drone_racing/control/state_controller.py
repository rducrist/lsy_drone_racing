from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """State controller with dynamic trajectory replanning and velocity-matched cubic splines."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = config.env.freq
        self._t_total = 20.0

        # initial observations
        self._pos = np.array(obs.get("pos"))
        self._quat = np.array(obs.get("quat"))
        self._vel = np.array(obs.get("vel", np.zeros(3)))
        self._gates_pos = np.array(obs.get("gates_pos"))
        self._gates_visited = np.array(obs.get("gates_visited"))
        self._gates_orientation = np.array(obs.get("gates_quat"))
        self._obstacles_pos = np.array(obs.get("obstacles_pos", []))
        self._target_gate = int(obs.get("target_gate", 0))

        # copy for replanning
        self._last_known_gates = np.array(self._gates_pos)
        self._needs_replanning = False

        # PD gains
        self._Kp = np.array([1.5, 1.5, 1.0])
        self._Kd = np.array([0.4, 0.4, 0.1])

        self._prev_pos_error = np.zeros(3)
        self._prev_time = 0.0

        # obstacle avoidance
        self._avoid_radius = 0.0
        self._avoid_gain = 0.0
        self._max_corr = np.array([1.0, 1.0, 1.0])

        self.SAFE_WAYPOINTS = np.array(
            [
                [0.0, 0.25, 0.50],  # around obstacle 1 (0.0, 0.75) → below/right path
                [1.4, 0.25, 1.00],  # around obstacle 2 (1.0, 0.25) → slightly under/right
                [-0.25, -0.15, 1.00],  # around obstacle 3 (-1.5, -0.25) → left/rear side
                [-0.5, -1.30, 1.00],  # around obstacle 4 (-0.5, -0.75) → below/left
            ]
        )

        # bookkeeping
        self._tick = 0
        self._finished = False
        self._actual_positions: list[np.ndarray] = []
        self._planned_trajectories: list[np.ndarray] = []  # store planned splines for visualization
        self._updated_gates_history: list[np.ndarray] = []  # store gate positions at each replan

        # build initial trajectory
        self._pos_current = np.array(self._pos)
        self._vel_current = np.array(self._vel)
        self._yaw_current = np.array(R.from_quat(self._quat).as_euler("xyz", degrees=False)[2])
        self._update_trajectory()

    # -------------------------------------------------------------------------
    def _update_trajectory(self):
        """Rebuild cubic spline trajectory from current state and known gates."""
        curr_pos = np.array(self._pos_current, dtype=float)
        curr_vel = np.array(self._vel_current, dtype=float)
        gates = np.array(self._last_known_gates, dtype=float)
        orientations = np.array(self._gates_orientation, dtype=float)

        if gates.size == 0:
            self._spline_x = lambda u: curr_pos[0]
            self._spline_y = lambda u: curr_pos[1]
            self._spline_z = lambda u: curr_pos[2]
            self._spline_yaw = lambda u: 0.0
            self._total_length = 1.0
            self._needs_replanning = False
            return

        APPROACH_DIST = 0.2
        waypoints = [curr_pos]
        yaw_points = [self._yaw_current]

        for i in range(self._target_gate, len(gates)):
            gate_pos = gates[i]
            rotation = R.from_quat(orientations[i])
            direction = rotation.apply([1, 0, 0])
            yaw = rotation.as_euler("xyz", degrees=False)[2]

            approach = gate_pos - APPROACH_DIST * direction
            departure = gate_pos + APPROACH_DIST * direction

            waypoints += [self.SAFE_WAYPOINTS[i], approach, gate_pos, departure]
            yaw_points += [yaw, yaw, yaw, yaw]

        # if len(waypoints) > 3 and self._target_gate < len(gates) -1:
        #     waypoints.insert(-3, np.array([-1.0, -1.5, 1.0]))
        #     yaw_points.insert(-3, yaw_points[-3])

        waypoints += [[-1.0, 0.0, 1.2]]
        yaw_points += [0]
        waypoints = np.array(waypoints)

        yaw_points = np.array(yaw_points)

        # --- compute cumulative segment lengths ---
        seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cum_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = cum_lengths[-1] if cum_lengths[-1] > 0 else 1.0
        self._total_length = total_length
        u_way = cum_lengths / total_length

        # --- adaptive total time ---
        # keep same average speed as original t_total
        avg_speed = total_length / self._t_total if hasattr(self, "_t_total") else 1.0
        self._t_total = total_length / avg_speed

        # --- Enforce velocity at start ---
        du_dt = 1.0 / total_length
        start_vel_u = curr_vel / du_dt  # convert to derivative wrt u

        bc_x = ((1, start_vel_u[0]), "natural")
        bc_y = ((1, start_vel_u[1]), "natural")
        bc_z = ((1, start_vel_u[2]), "natural")

        # Build splines
        self._spline_x = CubicSpline(u_way, waypoints[:, 0], bc_type=bc_x)
        self._spline_y = CubicSpline(u_way, waypoints[:, 1], bc_type=bc_y)
        self._spline_z = CubicSpline(u_way, waypoints[:, 2], bc_type=bc_z)
        self._spline_yaw = CubicSpline(u_way, yaw_points, bc_type="natural")

        self._total_length = total_length
        self._u_way = u_way
        self._needs_replanning = False
        self._prev_pos_error = np.zeros(3)
        self._prev_time = 0.0
        self._waypoints = waypoints

        num_samples = 200
        u_samples = np.linspace(0, 1, num_samples)
        traj = np.stack(
            [self._spline_x(u_samples), self._spline_y(u_samples), self._spline_z(u_samples)],
            axis=1,
        )
        self._planned_trajectories.append(traj)

        # store gates snapshot at this replan
        if self._last_known_gates is not None:
            self._updated_gates_history.append(np.array(self._last_known_gates))

    # -------------------------------------------------------------------------
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None):
        """PD tracking of cubic spline trajectory with velocity-matched start."""
        self._pos_current = np.array(obs.get("pos", np.zeros(3)))
        self._vel_current = np.array(obs.get("vel", np.zeros(3)))
        quat_current = np.array(obs.get("quat"))
        self._yaw_current = R.from_quat(quat_current).as_euler("xyz", degrees=False)[2]
        gates_pos = np.array(obs.get("gates_pos", self._last_known_gates))

        self._target_gate = np.array(obs.get("target_gate"))

        if not np.array_equal(gates_pos, self._last_known_gates):
            self._last_known_gates = gates_pos
            self._needs_replanning = True

        if self._needs_replanning:
            self._old_waypoints = np.array(self._waypoints).copy()
            self._update_trajectory()
            # breakpoint()
            self._tick = 0.0  # needs to be resetted

        # min (elapsed time und total time)

        t = min(self._tick / self._freq, self._t_total)
        u = np.clip(t / self._t_total, 0.0, 1.0)

        des_pos = np.array([self._spline_x(u), self._spline_y(u), self._spline_z(u)])

        des_yaw = float(self._spline_yaw(u))

        pos_error = des_pos - self._pos_current

        # simple XY obstacle avoidance
        if self._obstacles_pos is not None and len(self._obstacles_pos) > 0:
            for obs_pos in self._obstacles_pos:
                to_drone_xy = self._pos_current[:2] - obs_pos[:2]
                dist_xy = np.linalg.norm(to_drone_xy)
                if dist_xy < self._avoid_radius:
                    dir_xy = to_drone_xy / (dist_xy + 1e-6)
                    mag = self._avoid_gain * (1 - dist_xy / self._avoid_radius)
                    pos_error[:2] += mag * dir_xy

        # PD control
        dt = max(t - self._prev_time, 1e-6)
        d_error = (pos_error - self._prev_pos_error) / dt
        pos_correction = self._Kp * pos_error + self._Kd * d_error
        # pos_correction = np.clip(pos_correction, -self._max_corr, self._max_corr)

        self._prev_pos_error = pos_error
        self._prev_time = t

        controlled_pos = des_pos + pos_correction

        action = np.zeros(13, dtype=np.float32)
        action[:3] = controlled_pos
        action[9] = des_yaw

        if t >= self._t_total:
            self._finished = True

        return action

    # -------------------------------------------------------------------------
    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        if "pos" in obs:
            self._actual_positions.append(obs["pos"].copy())
        return self._finished

    # -------------------------------------------------------------------------
    def episode_callback(self):
        self.plot_desired_trajectory()
        self._tick = 0
        self._actual_positions.clear()

    # -------------------------------------------------------------------------
    def plot_desired_trajectory(self, num_samples: int = 500):
        """Plot all planned trajectories, updated gates, waypoints, and actual path."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # --- Plot all replanned trajectories ---
        if hasattr(self, "_planned_trajectories") and len(self._planned_trajectories) > 0:
            for i, traj in enumerate(self._planned_trajectories):
                alpha = 0.3 + 0.7 * (i + 1) / len(self._planned_trajectories)
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    traj[:, 2],
                    linestyle="--",
                    alpha=alpha,
                    label=f"Trajectory {i + 1}",
                )
            # highlight latest
            traj = self._planned_trajectories[-1]
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                color="blue",
                linewidth=2.0,
                label="Latest Planned Trajectory",
            )

        # --- Plot updated gate positions history ---
        if hasattr(self, "_updated_gates_history") and len(self._updated_gates_history) > 0:
            for i, gates in enumerate(self._updated_gates_history):
                ax.scatter(
                    gates[:, 0],
                    gates[:, 1],
                    gates[:, 2],
                    color="deepskyblue",
                    s=60,
                    marker="^",
                    alpha=0.6,
                    label=f"Updated Gates #{i + 1}"
                    if i == len(self._updated_gates_history) - 1
                    else None,
                )
            # latest gate update in darker color
            latest_gates = self._updated_gates_history[-1]
            ax.scatter(
                latest_gates[:, 0],
                latest_gates[:, 1],
                latest_gates[:, 2],
                color="blue",
                s=90,
                marker="^",
                label="Latest Gates",
            )

        # --- Plot original (initial) gates ---
        if hasattr(self, "_gates_pos") and self._gates_pos is not None:
            gates = np.array(self._gates_pos)
            if gates.size:
                ax.scatter(
                    gates[:, 0],
                    gates[:, 1],
                    gates[:, 2],
                    color="red",
                    s=80,
                    marker="*",
                    label="Initial Gates",
                )

        # --- Plot actual path ---
        if self._actual_positions:
            actual_positions = np.vstack(self._actual_positions)
            ax.plot(
                actual_positions[:, 0],
                actual_positions[:, 1],
                actual_positions[:, 2],
                color="green",
                linewidth=1.5,
                label="Actual Drone Path",
            )

        # --- Plot latest waypoints ---
        if hasattr(self, "_waypoints") and len(self._waypoints) > 0:
            wps = np.array(self._waypoints)
            ax.scatter(
                wps[:, 0],
                wps[:, 1],
                wps[:, 2],
                color="orange",
                s=40,
                marker="o",
                label="Latest Waypoints",
            )
            for i, p in enumerate(wps):
                ax.text(p[0], p[1], p[2], f"{i}", color="orange", fontsize=7)

        # --- Plot initial (old) waypoints ---
        if hasattr(self, "_old_waypoints") and len(self._old_waypoints) > 0:
            old_wps = np.array(self._old_waypoints)
            ax.scatter(
                old_wps[:, 0],
                old_wps[:, 1],
                old_wps[:, 2],
                color="purple",
                s=30,
                marker="x",
                label="Old Waypoints",
            )
            for i, p in enumerate(old_wps):
                ax.text(p[0], p[1], p[2], f"{i}", color="purple", fontsize=6)

        # --- Obstacles ---
        if self._obstacles_pos is not None and len(self._obstacles_pos) > 0:
            obstacles = np.array(self._obstacles_pos)
            ax.scatter(
                obstacles[:, 0],
                obstacles[:, 1],
                obstacles[:, 2],
                color="crimson",
                s=50,
                marker="x",
                label="Obstacles",
            )

        # --- Axes and layout ---
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Trajectory Evolution with Gate Updates and Waypoints")
        ax.legend(loc="best", fontsize=8)

        # Equal aspect ratio for 3D plot
        ranges = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        max_range = np.ptp(ranges) / 2.0
        mid_x, mid_y, mid_z = np.mean(ranges, axis=1)
        ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
        ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
        ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()
