"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

from lsy_drone_racing.utils.utils import draw_line

from scipy.spatial.transform import Rotation as R


if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)

def _gate_plane_corners(gate_pos: np.ndarray, rpy: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return 4 corners for two rectangles around the gate center.
    Ohne Rotation → Weltkoordinaten direkt. Mit Rotation → Lokale Koordinaten → Rotieren → Weltkoordinaten.
    """
   # First rectangle (smaller, 0.27m side)
    y_min1 = -0.135
    y_max1 = 0.135
    z_min1 = -0.135
    z_max1 = 0.135
    local_corners1 = np.array([[0, y_min1, z_min1],
                               [0, y_max1, z_min1],
                               [0, y_max1, z_max1],
                               [0, y_min1, z_max1]], dtype=float)
    # Second rectangle (larger, 0.88m side)
    y_min2 = -0.44
    y_max2 = 0.44
    z_min2 = -0.44
    z_max2 = 0.44
    local_corners2 = np.array([[0, y_min2, z_min2],
                               [0, y_max2, z_min2],
                               [0, y_max2, z_max2],
                               [0, y_min2, z_max2]], dtype=float)
    # Rotation matrix from rpy
    rot = R.from_euler('xyz', rpy)
    rot_matrix = rot.as_matrix()
    
    # Rotate and translate to world coordinates
    corners1 = (rot_matrix @ local_corners1.T).T + gate_pos
    corners2 = (rot_matrix @ local_corners2.T).T + gate_pos

    return corners1, corners2

def _draw_gate_plane_lines(env_unwrapped, gates, config, color=(0.0, 0.8, 0.0, 1.0), min_size: float = 2.0, max_size: float = 2.0):
    """Draw rectangle edges for two rectangles per gate using draw_line.
    env_unwrapped: env.unwrapped passed to draw_line as in existing calls.
    gates: iterable of [x,y,z] gate centers.
    config: config object to access gate rpy.
    """
    if gates is None:
        return
    rgba = np.array(color, dtype=float)
    edges_idx = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, g in enumerate(gates):
        gate_config = config.env.track.gates[i]
        rpy = gate_config['rpy']
        corners1, corners2 = _gate_plane_corners(np.array(g), rpy)
        for corners in [corners1, corners2]:
            for a, b in edges_idx:
                pts = np.vstack([corners[a], corners[b]])  # shape (2,3)
                try:
                    draw_line(env=env_unwrapped, points=pts, rgba=rgba, min_size=min_size, max_size=max_size)
                except TypeError:
                    # tolerant fallback if signature differs
                    try:
                        draw_line(points=pts, rgba=rgba, min_size=min_size, max_size=max_size)
                    except Exception:
                        pass
                except Exception:
                    pass


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            # Convert to a buffer that meets XLA's alginment restrictions to prevent warnings. See
            # https://github.com/jax-ml/jax/discussions/6055
            # Tracking issue:
            # https://github.com/jax-ml/jax/issues/29810
            action = np.asarray(jp.asarray(action), copy=True)

            obs, reward, terminated, truncated, info = env.step(action)
            # Update the controller internal state and models.
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )

            if controller._predicted_traj is not None:
                draw_line(
                    env=env.unwrapped,
                    points=controller._predicted_traj,
                    rgba=np.array([0.0,0.3,1.0,0.7]),
                    min_size=2.0,
                    max_size=3.0
                )


            

            if controller.traj_viz is not None:
                draw_line(
                    env=env.unwrapped,
                    points=controller.traj_viz,
                    rgba=np.array([52, 200, 72, 0.8]),
                    min_size=2.0,
                    max_size=3.0
                )

             # Draw gate plane edges
            _draw_gate_plane_lines(env.unwrapped, obs.get("gates_pos"),config)


            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            if config.sim.render:  # Render the sim if selected.
                if ((i * fps) % config.env.freq) < fps:
                    env.render()
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
