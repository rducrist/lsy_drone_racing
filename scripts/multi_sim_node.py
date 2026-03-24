from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import fire
import gymnasium
import numpy as np
import rclpy
from drone_racing_msgs.msg import Action, EpisodeEnd, EpisodeReset, Observations, StepResult
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from rclpy.node import Node

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"

def flatten_dict(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v).reshape(-1) for k, v in obs.items()}


class MultiSimNode(Node):
    def __init__(self, config, external_drone_id: int = 0, action_timeout: float = 5.0):
        super().__init__("multi_sim")
        self.config = config
        self.external_drone_id = external_drone_id
        self.action_timeout = action_timeout

        self.episode_id = 0
        self.step_id = 0
        self.pending_action = None

        self.reset_pub = self.create_publisher(EpisodeReset, "/race/reset", 10)
        self.observation_pub = self.create_publisher(Observations, "/race/observations", 10)
        self.step_result_pub = self.create_publisher(StepResult, "/race/step_result", 10)
        self.episode_end_pub = self.create_publisher(EpisodeEnd, "/race/episode_end", 10)
        self.create_subscription(Action, "/race/action", self.on_action, 10)

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

    def on_action(self, msg: Action):
        if msg.episode_id != self.episode_id:
            return
        if msg.step_id != self.step_id:
            return
        if msg.drone_id != self.external_drone_id:
            return
        self.pending_action = np.asarray(msg.action, dtype=np.float32)

    def wait_for_action(self) -> np.ndarray:
        deadline = time.monotonic() + self.action_timeout
        while rclpy.ok():
            if self.pending_action is not None:
                action = self.pending_action
                self.pending_action = None
                return action
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for action at episode={self.episode_id}, step={self.step_id}"
                )
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))
        raise RuntimeError("ROS shutdown while waiting for action")
    
    def publish_reset(self, obs: dict[str, Any]):
        flat = flatten_dict(obs)
        msg = EpisodeReset()
        msg.episode_id = self.episode_id
        msg.external_drone_id = self.external_drone_id
        msg.pos = flat["pos"].tolist()
        msg.quat = flat["quat"].tolist()
        msg.vel = flat["vel"].tolist()
        msg.ang_vel = flat["ang_vel"].tolist()
        msg.target_gate = np.asarray(obs["target_gate"]).reshape(-1).astype(np.int32).tolist()
        msg.gates_pos = flat["gates_pos"].tolist()
        msg.gates_quat = flat["gates_quat"].tolist()
        msg.gates_visited = np.asarray(obs["gates_visited"]).reshape(-1).astype(bool).tolist()
        msg.obstacles_pos = flat["obstacles_pos"].tolist()
        self.reset_pub.publish(msg)

    def publish_observations(self, obs: dict[str, Any]):
        flat = flatten_dict(obs)
        msg = Observations()
        msg.episode_id = self.episode_id
        msg.step_id = self.step_id
        msg.external_drone_id = self.external_drone_id
        msg.pos = flat["pos"].tolist()
        msg.quat = flat["quat"].tolist()
        msg.vel = flat["vel"].tolist()
        msg.ang_vel = flat["ang_vel"].tolist()
        msg.target_gate = np.asarray(obs["target_gate"]).reshape(-1).astype(np.int32).tolist()
        msg.gates_pos = flat["gates_pos"].tolist()
        msg.gates_quat = flat["gates_quat"].tolist()
        msg.gates_visited = np.asarray(obs["gates_visited"]).reshape(-1).astype(bool).tolist()
        msg.obstacles_pos = flat["obstacles_pos"].tolist()
        self.observation_pub.publish(msg)

    def publish_step_result(
            self,
            action: np.ndarray,
            obs: dict[str, Any],
            reward: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray
    ):
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
        msg.reward = np.asarray(reward).reshape(-1).astype(np.float32).tolist()
        msg.terminated = np.asarray(terminated).reshape(-1).astype(bool).tolist()
        msg.truncated = np.asarray(truncated).reshape(-1).astype(bool).tolist()
        self.step_result_pub.publish(msg)

    def publish_episode_end(self, curr_time: float, obs: dict[str, Any]):
        msg = EpisodeEnd()
        msg.episode_id = self.episode_id
        msg.curr_time = float(curr_time)
        msg.finished = bool(np.all(np.asarray(obs["target_gate"]) == -1))
        self.episode_end_pub.publish(msg)

    # The main loop
    def run(self, n_runs: int = 1):
        for _ in range(n_runs):
            self.episode_id += 1
            self.step_id = -1
            self.pending_action = None

            obs, info = self.env.reset()
            self.publish_reset(obs)

            i = 0
            fps = 60
            curr_time = 0.0

            while(True):
                curr_time = i / self.config.env.freq
                self.step_id = i
                self.publish_observations(obs)

                action = self.wait_for_action()
                obs, reward, terminated, truncated, info = self.env.step(action)

                self.publish_step_result(action, obs, reward, terminated, truncated)

                done = terminated | truncated

                if self.config.sim.gui:
                    if ((i * fps) % self.config.env.freq) < fps:
                        self.env.render()
                i += 1
                if done:
                    break
            
            self.publish_episode_end(curr_time, obs)

        self.env.close()

def simulate(
    config: str = "multi_level0.toml",
    n_runs: int = 1,
    gui: bool | None = None,
    external_drone_id: int = 0,
    action_timeout: float = 100.0,
):
    config = load_config(CONFIG_ROOT / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui

    rclpy.init()
    node = MultiSimNode(
        config=config,
        external_drone_id=external_drone_id,
        action_timeout=action_timeout
    )

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
