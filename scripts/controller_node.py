from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
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
from rclpy.node import Node

from lsy_drone_racing.control.multi_agent_wrapper import wrap_controller
from lsy_drone_racing.utils import load_config, load_controller


class ControllerNode(Node):
    """Creates a generic controller node."""
    def __init__(self, controller_path: Path, config_path: Path, drone_id: int):
        """Initializes the controller node.
        
        Args:
            controller_path: path to the controller
            config_path: path to the environment config
            drone_id: id of the drone
        """
        super().__init__(f"controller_{drone_id}")
        self.drone_id = drone_id
        self.config = load_config(config_path)
        self.config.env.freq = self.config.env.kwargs[0]["freq"]

        self.controller_cls = wrap_controller(load_controller(controller_path))
        self.controller = None
        self.last_action = None
        self.finished = False

        self.action_pub = self.create_publisher(
            Action, 
            f"/race/drone_{self.drone_id}/action", 
            10)
        self.create_subscription(EpisodeReset, "/race/reset", self.on_reset, 10)
        self.create_subscription(Observations, "/race/observations", self.on_observations, 10)
        self.create_subscription(StepResult, "/race/step_result", self.on_step_result, 10)
        self.create_subscription(EpisodeEnd, "/race/episode_end", self.on_episode_end, 10)
        self.create_subscription(RaceEnd, "/race/race_end", self.on_race_end, 10)
        self.get_logger().info(f"Finished Init Controller {self.drone_id}")


    def reshape_observations(self, msg: EpisodeReset | Observations | StepResult) -> dict[str, Any]:
        """Reshaping the obs as dict."""
        # self._update_n_drones_from_msg(msg)
        obs = {
            "pos": np.asarray(msg.pos, dtype=np.float32).reshape(-1, 3),
            "quat": np.asarray(msg.quat, dtype=np.float32).reshape(-1, 4),
            "vel": np.asarray(msg.vel, dtype=np.float32).reshape(-1, 3),
            "ang_vel": np.asarray(msg.ang_vel, dtype=np.float32).reshape(-1, 3),
            "target_gate": np.asarray(msg.target_gate, dtype=np.int32).reshape(-1),
            "gates_pos": np.asarray(msg.gates_pos, dtype=np.float32).reshape(-1, 4, 3),
            "gates_quat": np.asarray(msg.gates_quat, dtype=np.float32).reshape(-1, 4, 4),
            "gates_visited": np.asarray(msg.gates_visited, dtype=bool).reshape(-1, 4),
            "obstacles_pos": np.asarray(msg.obstacles_pos, dtype=np.float32).reshape(-1, 4, 3),
            "obstacles_visited": np.asarray(msg.obstacles_visited, dtype=bool).reshape(-1, 4),
        }
        return obs
    
    
    def on_reset(self, msg: EpisodeReset):
        """Creates controller instance."""
        obs = self.reshape_observations(msg)
        info = {"rank" : self.drone_id}
        self.controller = self.controller_cls(obs, info, self.config)
        self.get_logger().info("Reset succeeded")

    def on_observations(self, msg: Observations):
        """Computes control and publishes action."""
        if self.controller is None:
            return
        obs = self.reshape_observations(msg)
        info = {"rank" : self.drone_id}
        action = self.controller.compute_control(obs, info)
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
        """Populates controller step callback."""
        if self.controller is None or self.last_action is None:
            return
        obs = self.reshape_observations(msg)
        reward = np.asarray(msg.reward, dtype=np.float32)
        terminated = np.asarray(msg.terminated, dtype=bool)
        truncated = np.asarray(msg.truncated, dtype=bool)
        info = {"rank" : self.drone_id}
        self.controller.step_callback(
            self.last_action,
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def on_episode_end(self, msg:EpisodeEnd):
        """Resets the episode."""
        # if self.controller is None:
        #     self.finished = True
        #     return
        self.controller.episode_callback()
        self.controller.episode_reset()

    def on_race_end(self, msg:RaceEnd):
        """Ends the race."""
        self.finished = True

def main(
        drone_id: int = 0,
        controller: str = "attitude_controller.py",
        config: str = "multi_level0.toml"
):
    """Main control loop."""
    project_root = Path(__file__).resolve().parents[1]

    rclpy.init()
    node = ControllerNode(
        controller_path=project_root / f"lsy_drone_racing/control/{controller}",
        config_path=project_root / "config" / config,
        drone_id=drone_id,
    )

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
