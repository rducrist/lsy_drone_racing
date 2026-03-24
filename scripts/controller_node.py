from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from lsy_drone_racing.utils import load_config, load_controller
from drone_racing_msgs.msg import Action, EpisodeEnd, EpisodeReset, Observations, StepResult
import fire


class ControllerNode(Node):
    def __init__(self, controller_path: Path, config_path: Path, drone_id: int, n_drones: int):
        super().__init__(f"controller_{drone_id}")
        self.drone_id = drone_id
        self.n_drones = n_drones
        self.config = load_config(config_path)
        self.config.env.freq = self.config.env.kwargs[0]["freq"]

        self.controller_cls = load_controller(controller_path)
        self.controller = None
        self.last_action = None

        self.action_pub = self.create_publisher(
            Action, 
            f"/race/drone_{self.drone_id}/action", 
            10)
        self.create_subscription(EpisodeReset, "/race/reset", self.on_reset, 10)
        self.create_subscription(Observations, "/race/observations", self.on_observations, 10)
        self.create_subscription(StepResult, "/race/step_result", self.on_step_result, 10)
        self.create_subscription(EpisodeEnd, "/race/episode_end", self.on_episode_end, 10)
        self.get_logger().info(f"Finished Init Controller {self.drone_id}")

    def _update_n_drones_from_msg(self, msg) -> None:
        if len(msg.pos) % 3 != 0:
            raise ValueError(f"Position vector has invalid length {len(msg.pos)}")
        n_drones = len(msg.pos) // 3
        if n_drones <= 0:
            raise ValueError("Received message without any drone positions")
        if self.drone_id >= n_drones:
            raise ValueError(f"Configured drone_id={self.drone_id} but message only has {n_drones} drones")
        self.n_drones = n_drones

    def reshape_observations(self, msg):
        self._update_n_drones_from_msg(msg)
        obs = {
            "pos": np.asarray(msg.pos, dtype=np.float32).reshape(self.n_drones, 3),
            "quat": np.asarray(msg.quat, dtype=np.float32).reshape(self.n_drones, 4),
            "vel": np.asarray(msg.vel, dtype=np.float32).reshape(self.n_drones, 3),
            "ang_vel": np.asarray(msg.ang_vel, dtype=np.float32).reshape(self.n_drones, 3),
            "target_gate": np.asarray(msg.target_gate, dtype=np.int32).reshape(self.n_drones),
            "gates_pos": np.asarray(msg.gates_pos, dtype=np.float32),
            "gates_quat": np.asarray(msg.gates_quat, dtype=np.float32),
            "gates_visited": np.asarray(msg.gates_visited, dtype=bool),
            "obstacles_pos": np.asarray(msg.obstacles_pos, dtype=np.float32),
        }
        return obs
    
    def drone_obs(self, obs):
        return {
            "pos": obs["pos"][self.drone_id],
            "quat": obs["quat"][self.drone_id],
            "vel": obs["vel"][self.drone_id],
            "ang_vel": obs["ang_vel"][self.drone_id],
            "target_gate": obs["target_gate"][self.drone_id],
            "gates_pos": obs["gates_pos"],
            "gates_quat": obs["gates_quat"],
            "gates_visited": obs["gates_visited"],
            "obstacles_pos": obs["obstacles_pos"],
        }
    
    def on_reset(self, msg: EpisodeReset):
        obs = self.reshape_observations(msg)
        info = {}
        self.controller = self.controller_cls(self.drone_obs(obs), info, self.config)
        self.get_logger().info("Reset succeeded")

    def on_observations(self, msg: Observations):
        if self.controller is None:
            return
        obs = self.reshape_observations(msg)
        info = {}
        action = self.controller.compute_control(self.drone_obs(obs), info)
        self.last_action = np.asanyarray(action, dtype=np.float32)
        self.get_logger().info("Received Observations")


        out = Action()
        out.episode_id = msg.episode_id
        out.step_id = msg.step_id
        out.drone_id = self.drone_id
        out.action = self.last_action.tolist()
        self.action_pub.publish(out)
        self.get_logger().info("Action sent")


    def on_step_result(self, msg: StepResult):
        if self.controller is None or self.last_action is None:
            return
        obs = self.reshape_observations(msg)
        reward = np.asarray(msg.reward, dtype=np.float32)[0]
        terminated = bool(np.asarray(msg.terminated)[0])
        truncated = bool(np.asarray(msg.truncated)[0])
        info = {}
        self.controller.step_callback(
            self.last_action,
            self.drone_obs(obs),
            reward,
            terminated,
            truncated,
            info,
        )

    def on_episode_end(self, msg: EpisodeEnd):
        if self.controller is None:
            return
        self.controller.episode_callback()
        self.controller.episode_reset()
        self.get_logger().info("Episode ended")


def main(
        drone_id: int = 0,
        controller: str = "attitude_controller.py",
        config: str = "multi_level0.toml"
):
    project_root = Path(__file__).resolve().parents[1]

    rclpy.init()
    node = ControllerNode(
        controller_path=project_root / f"lsy_drone_racing/control/{controller}",
        config_path=project_root / "config/multi_level0.toml",
        drone_id=drone_id,
        n_drones=1
    )

    try:
        rclpy.spin(node)

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)