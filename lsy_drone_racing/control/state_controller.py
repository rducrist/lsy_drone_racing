from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =====================================================================
# PMM TRAJECTORY WRAPPER
# =====================================================================
class PMMTrajectory:
    """Wrapper for PMM-generated time-optimal trajectory CSV."""

    def __init__(self, path: str):
        cols = ["t", "p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z"]
        traj = pd.read_csv(path, names=cols, comment="#")

        self.t = traj["t"].to_numpy()
        self.pos = traj[["p_x", "p_y", "p_z"]].to_numpy()
        self.vel = traj[["v_x", "v_y", "v_z"]].to_numpy()
        self.acc = traj[["a_x", "a_y", "a_z"]].to_numpy()

        # Cubic splines for smooth interpolation
        self.spline_pos = [CubicSpline(self.t, self.pos[:, i]) for i in range(3)]
        self.spline_vel = [CubicSpline(self.t, self.vel[:, i]) for i in range(3)]
        self.spline_acc = [CubicSpline(self.t, self.acc[:, i]) for i in range(3)]

        self.t_final = float(self.t[-1])

    def get_desired(self, t: float):
        """Return desired position, velocity, and acceleration at time t."""
        t = np.clip(t, self.t[0], self.t_final)
        p = np.array([s(t) for s in self.spline_pos])
        v = np.array([s(t) for s in self.spline_vel])
        a = np.array([s(t) for s in self.spline_acc])
        return p, v, a


# =====================================================================
# STATE CONTROLLER
# =====================================================================
class StateController(Controller):
    """Tracks a precomputed PMM trajectory using PD + velocity feedforward."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        # frequency
        self._freq = config.env.freq
        self._tick = 0

        # PD gains
        self._Kp = np.array([2.0, 2.0, 2.5])
        self._Kd = np.array([0.8, 0.8, 0.5])

        # feedforward scaling (optional)
        self._feedforward_gain = 1.0

        # finished flag
        self._finished = False

        # load PMM trajectory
        self._traj = PMMTrajectory("lsy_drone_racing/trajectories/sampled_trajectory.csv")
        self._t_final = self._traj.t_final

        # logs for visualization
        self._actual_positions: list[np.ndarray] = []
        self._desired_positions: list[np.ndarray] = []
        self._desired_velocities: list[np.ndarray] = []

        print(f"Loaded PMM trajectory with {len(self._traj.t)} samples, T={self._t_final:.2f}s")

    # -----------------------------------------------------------------
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None):
        """Compute PD tracking control at current simulation step."""
        t = self._tick / self._freq

        # current state
        pos = np.array(obs["pos"])
        vel = np.array(obs["vel"])
        quat = np.array(obs["quat"])
        yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[2]

        # desired state
        des_pos, des_vel, des_acc = self._traj.get_desired(t)

        # PD + velocity feedforward control
        pos_error = des_pos - pos
        vel_error = des_vel - vel
        pos_correction = self._Kp * pos_error + self._Kd * vel_error
        # pos_correction += self._feedforward_gain * des_acc * 0.05  # small feedforward accel term

        controlled_pos = des_pos + pos_correction

        # Compute yaw from velocity direction
        yaw_des = np.arctan2(des_vel[1], des_vel[0]) if np.linalg.norm(des_vel[:2]) > 1e-3 else yaw

        # Build action vector
        action = np.zeros(13, dtype=np.float32)
        action[:3] = controlled_pos
        action[9] = yaw_des

        # store data
        self._actual_positions.append(pos)
        self._desired_positions.append(des_pos)
        self._desired_velocities.append(des_vel)

        # update time
        self._tick += 1
        if t >= self._t_final:
            self._finished = True

        return action

    # -----------------------------------------------------------------
    def step_callback(self, action, obs, reward, terminated, truncated, info):
        """Called each simulation step."""
        return self._finished

    # -----------------------------------------------------------------
    def episode_callback(self):
        """Called at the end of the episode (for plotting)."""
        self.plot_trajectory()

    # -----------------------------------------------------------------
    def plot_trajectory(self):
        """Plot desired vs actual 3D trajectory."""
        if not self._actual_positions:
            return

        actual = np.vstack(self._actual_positions)
        desired = np.vstack(self._desired_positions)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], "b--", label="Desired (PMM)")
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], "g", label="Actual (Tracked)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("PMM Trajectory Tracking")
        ax.legend()
        plt.tight_layout()
        plt.show()
