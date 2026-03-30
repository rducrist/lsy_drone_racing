from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import fire
import gymnasium
import numpy as np
import rclpy
from drone_racing_msgs.msg import (
    Action,
    EpisodeEnd,
    EpisodeReset,
    Observations,
    RaceEnd,
    StepResult,
)
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from rclpy.node import Node

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"

def flatten_dict(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Observations need to be flattened for easy handling."""
    return {k: np.asarray(v).reshape(-1) for k, v in obs.items()}


class MultiSimNode(Node):
    """Create ROS node to wrap the simulator into."""
    def __init__(self, config: dict, action_timeout: float = 5.0):
        """Initializes the simulator node.
        
        Args:
            config: The configuration of the environment.
            action_timeout: Timeout limit for receiving actions from either controller.
        """
        super().__init__("multi_sim")
        self.config = config
        self.action_timeout = action_timeout

        self.episode_id = 0
        self.step_id = 0
        self.pending_actions: dict[int, np.ndarray] = {}

        self.env = gymnasium.make(
            "MultiDroneRacing-v0",
            freq=config.env.kwargs[0]["freq"],
            sim_config=config.sim,
            track=config.env.track,
            sensor_range=config.env.kwargs[0]["sensor_range"],
            control_mode=config.env.kwargs[0]["control_mode"],
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
        )
        config.env.freq = config.env.kwargs[0]["freq"]
        self.env = JaxToNumpy(self.env)
        self.n_drones = self.env.unwrapped.sim.n_drones
        self.n_worlds = self.env.unwrapped.sim.n_worlds
        self.episode_times: list[float] = []
        self.episode_finished: list[bool] = []

        self.reset_pub = self.create_publisher(EpisodeReset, "/race/reset", 10)
        self.observation_pub = self.create_publisher(Observations, "/race/observations", 10)
        self.step_result_pub = self.create_publisher(StepResult, "/race/step_result", 10)
        self.episode_end_pub = self.create_publisher(EpisodeEnd, "/race/episode_end", 10)
        self.race_end_pub = self.create_publisher(RaceEnd, "/race/race_end", 10)

        self.action_subs = []
        for drone_id in range(self.n_drones):
            topic = f"/race/drone_{drone_id}/action"
            self.action_subs.append(
                self.create_subscription(
                    Action, topic, lambda msg, drone_id=drone_id: self.on_action(msg, drone_id), 10
                )
            )

    def on_action(self, msg: Action, drone_id: int):
        """Check if received action matches the timestamps and drone id."""
        if msg.episode_id != self.episode_id:
            return
        if msg.step_id != self.step_id:
            return
        if msg.drone_id != drone_id:
            return
        self.pending_actions[drone_id] = np.asarray(msg.action, dtype=np.float32)

    def wait_for_actions(self) -> np.ndarray:
        """Wait for incoming action from all controllers."""
        deadline = time.monotonic() + self.action_timeout
        required = set(range(self.n_drones))

        while rclpy.ok():
            if required.issubset(self.pending_actions):
                actions = np.stack(
                    [self.pending_actions[i] for i in range(self.n_drones)], axis=0
                ).astype(np.float32)
                self.pending_actions.clear()
                return actions

            # Check if any controller timed out
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(required - set(self.pending_actions))
                raise TimeoutError(
                    f"Timed out waiting for actions from drones {missing} "
                    f"at episode={self.episode_id}, step={self.step_id}"
                )
            # Trigger action callback once
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))

        raise RuntimeError("ROS shutdown while waiting for action")

    def publish_reset(self, obs: dict[str, Any]):
        """Publish observations on reset."""
        flat = flatten_dict(obs)
        msg = EpisodeReset()
        msg.episode_id = self.episode_id
        msg.pos = flat["pos"].tolist()
        msg.quat = flat["quat"].tolist()
        msg.vel = flat["vel"].tolist()
        msg.ang_vel = flat["ang_vel"].tolist()
        msg.target_gate = np.asarray(obs["target_gate"]).reshape(-1).astype(np.int32).tolist()
        msg.gates_pos = flat["gates_pos"].tolist()
        msg.gates_quat = flat["gates_quat"].tolist()
        msg.gates_visited = np.asarray(obs["gates_visited"]).reshape(-1).astype(bool).tolist()
        msg.obstacles_pos = flat["obstacles_pos"].tolist()
        msg.obstacles_visited= flat["obstacles_visited"].tolist()
        self.reset_pub.publish(msg)

    def publish_observations(self, obs: dict[str, Any]):
        """Publish observations."""
        flat = flatten_dict(obs)
        msg = Observations()
        msg.episode_id = self.episode_id
        msg.step_id = self.step_id
        msg.pos = flat["pos"].tolist()
        msg.quat = flat["quat"].tolist()
        msg.vel = flat["vel"].tolist()
        msg.ang_vel = flat["ang_vel"].tolist()
        msg.target_gate = np.asarray(obs["target_gate"]).reshape(-1).astype(np.int32).tolist()
        msg.gates_pos = flat["gates_pos"].tolist()
        msg.gates_quat = flat["gates_quat"].tolist()
        msg.gates_visited = np.asarray(obs["gates_visited"]).reshape(-1).astype(bool).tolist()
        msg.obstacles_pos = flat["obstacles_pos"].tolist()
        msg.obstacles_visited= flat["obstacles_visited"].tolist()

        self.observation_pub.publish(msg)

    def publish_step_result(
        self,
        action: np.ndarray,
        obs: dict[str, Any],
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ):
        """Publish the step result."""
        flat = flatten_dict(obs)
        msg = StepResult()
        msg.episode_id = self.episode_id
        msg.step_id = self.step_id
        msg.action = np.asarray(action).reshape(-1).astype(np.float32).tolist()

        msg.pos = flat["pos"].tolist()
        msg.quat = flat["quat"].tolist()
        msg.vel = flat["vel"].tolist()
        msg.ang_vel = flat["ang_vel"].tolist()
        msg.target_gate = np.asarray(obs["target_gate"]).reshape(-1).astype(np.int32).tolist()
        msg.gates_pos = flat["gates_pos"].tolist()
        msg.gates_quat = flat["gates_quat"].tolist()
        msg.gates_visited = np.asarray(obs["gates_visited"]).reshape(-1).astype(bool).tolist()
        msg.obstacles_pos = flat["obstacles_pos"].tolist()
        msg.obstacles_visited= flat["obstacles_visited"].tolist()

        msg.reward = np.asarray(reward).reshape(-1).astype(np.float32).tolist()
        msg.terminated = np.asarray(terminated).reshape(-1).astype(bool).tolist()
        msg.truncated = np.asarray(truncated).reshape(-1).astype(bool).tolist()
        self.step_result_pub.publish(msg)

    def publish_episode_end(self, curr_time: float, obs: dict[str, Any]):
        """Publishes the episode results of the drones."""
        finished = bool(np.all(np.asarray(obs["target_gate"]) == -1))
        msg = EpisodeEnd()
        msg.episode_id = self.episode_id
        msg.curr_time = float(curr_time)
        msg.finished = finished
        self.episode_end_pub.publish(msg)

    def publish_race_end(self):
        """Signals the controllers that the race is ended."""
        msg = RaceEnd()
        msg.race_finished = True
        self.race_end_pub.publish(msg)
        
    # The main loop
    def run(self, n_runs: int = 1):
        """Runs the simulator loop."""
        for _ in range(n_runs):
            finish_times = [None] * self.n_drones
            self.episode_id += 1
            self.step_id = -1
            self.pending_actions.clear()

            obs, info = self.env.reset()
            self.publish_reset(obs)

            i = 0
            fps = 60
            curr_time = 0.0

            while True:
                curr_time = i / self.config.env.freq
                self.step_id = i
                self.publish_observations(obs)

                actions = self.wait_for_actions()
                obs, reward, terminated, truncated, info = self.env.step(actions)

                self.publish_step_result(actions, obs, reward, terminated, truncated)

                done = terminated | truncated

                if self.config.sim.gui:
                    if ((i * fps) % self.config.env.freq) < fps:
                        self.env.render()
                i += 1

                for drone_id in range(self.n_drones):
                    if finish_times[drone_id] is None and obs["target_gate"][drone_id]==-1:
                        finish_times[drone_id] = curr_time

                if done:
                    break

            self.publish_episode_end(curr_time, obs)
            self.episode_times.append(finish_times.copy())
            self.episode_finished.append(all(t is not None for t in finish_times))

            self.get_logger().info(
                f"Episode {self.episode_id}: {self._format_drone_finish_times(finish_times)}"
            )

        self.publish_race_end()



        if self.episode_times:
            summary = " | ".join(
                [
                    f"ep {idx + 1}: {self._format_drone_finish_times(finish_times)}"
                    for idx, finish_times in enumerate(self.episode_times)
                ]
            )
            self.get_logger().info(f"Race summary: {summary}")


        self.env.close()

    def _format_drone_finish_times(self, finish_times: list[float | None]) -> str:
        parts = []
        for drone_id, t in enumerate(finish_times):
            if t is None:
                parts.append(f"drone_{drone_id}=DNF")
            else:
                parts.append(f"drone_{drone_id}={t:.3f}s")
        return ", ".join(parts)



def simulate(
    config: str = "multi_level0.toml",
    n_runs: int = 2,
    gui: bool | None = None,
    action_timeout: float = 10.0,
):
    """Run the simulator node."""
    config = load_config(CONFIG_ROOT / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui

    rclpy.init()
    node = MultiSimNode(config=config, action_timeout=action_timeout)

    try:
        node.run(n_runs)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
