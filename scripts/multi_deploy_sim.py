"""Simulate the drones of a multi-drone deploy config with crazyflow_itl.

Each drone is served as a Crazyflie over UDP at udp://127.0.0.1:19850 + rank and broadcasts its pose
on TF like the motion capture system. Set ``deploy.deploy_in_sim = true`` so the host connects to
the simulated drones.

Usage:

    python scripts/multi_deploy_sim.py --config multi_level2.toml --render

"""

from pathlib import Path

import fire
import jax.numpy as jnp
import rclpy
from crazyflow.sim import Sim
from crazyflow_itl import RealTimePlugin, RenderPlugin, run
from crazyflow_itl.crazyflie import CrazyfliePlugin
from crazyflow_itl.ros import TFPlugin
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config


def main(config: str = "multi_level2.toml", render: bool = False):
    """Run the simulated drones in real time until interrupted.

    Args:
        config: Configuration file name, assumed to be in ``config/``.
        render: Whether to show the sim in the MuJoCo viewer.
    """
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    drones = cfg.env.track.drones
    sim = Sim(
        n_drones=len(drones),
        drone=cfg.sim.drone,
        control="force_torque",
        dynamics="first_principles",
        freq=cfg.sim.freq,
    )
    pos = jnp.asarray([[d["pos"] for d in drones]], dtype=jnp.float32, device=sim.device)
    quat = jnp.asarray(
        [R.from_euler("xyz", [d["rpy"] for d in drones]).as_quat()],
        dtype=jnp.float32,
        device=sim.device,
    )
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos, quat=quat))
    rclpy.init()
    node = rclpy.create_node("crazyflow_itl")
    names = [f"cf{d['id']}" for d in cfg.deploy.drones]  # Must match the mocap frame names
    plugins = [TFPlugin(node, names=names), RealTimePlugin(), CrazyfliePlugin()]

    if render:
        plugins.insert(0, RenderPlugin())
    try:
        run(sim, plugins)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
