"""Analyze logged multi-drone simulation runs.

Run as:

    $ python scripts/analyze_multi_sim.py --input_path logs/multi_sim

This script expects `.npz` files produced by `scripts/multi_sim.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _resolve_log_files(input_path: str) -> list[Path]:
    """Return all `.npz` logs under the given path."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz logs found in: {path}")
    return files


def _drone_labels(data: np.lib.npyio.NpzFile) -> list[str]:
    """Get readable labels for each drone."""
    if "controller_names" in data:
        return [str(name) for name in data["controller_names"].tolist()]
    n_drones = data["pos"].shape[1]
    return [f"drone_{i}" for i in range(n_drones)]


def _plot_trajectory(data: np.lib.npyio.NpzFile, output_file: Path):
    """Save the xyz trajectory plots for all drones."""
    t = data["t"]
    pos = data["pos"]
    labels = _drone_labels(data)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
    coord_labels = ("x", "y", "z")
    for axis_idx, coord in enumerate(coord_labels):
        ax = axes[axis_idx]
        for drone_idx, label in enumerate(labels):
            ax.plot(t, pos[:, drone_idx, axis_idx], label=label)
        ax.set_ylabel(f"{coord} [m]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    xy_ax = axes[3]
    for drone_idx, label in enumerate(labels):
        xy_ax.plot(pos[:, drone_idx, 0], pos[:, drone_idx, 1], label=label)
    xy_ax.set_xlabel("x [m]")
    xy_ax.set_ylabel("y [m]")
    xy_ax.set_title("XY trajectory")
    xy_ax.grid(True, alpha=0.3)
    xy_ax.legend(loc="best")
    fig.suptitle("Drone trajectories")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def _plot_wrench(data: np.lib.npyio.NpzFile, output_file: Path):
    """Save the force and torque plots for all drones."""
    t = data["t"]
    force = data["force"]
    torque = data["torque"]
    labels = _drone_labels(data)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    force_labels = ("Fx", "Fy", "Fz")
    torque_labels = ("Tx", "Ty", "Tz")

    for axis_idx, coord in enumerate(force_labels):
        ax = axes[0, axis_idx]
        for drone_idx, label in enumerate(labels):
            ax.plot(t, force[:, drone_idx, axis_idx], label=label)
        ax.set_ylabel(f"{coord} [N]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    for axis_idx, coord in enumerate(torque_labels):
        ax = axes[1, axis_idx]
        for drone_idx, label in enumerate(labels):
            ax.plot(t, torque[:, drone_idx, axis_idx], label=label)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{coord} [Nm]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("Force and torque")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def analyze(
    input_path: str = "logs/multi_sim",
    output_dir: str | None = None,
):
    """Generate trajectory and force/torque plots from simulation logs.

    Args:
        input_path: A `.npz` file or a directory containing `.npz` logs.
        output_dir: Directory used to save figures. Defaults to `<input_path>/plots` for
            directories or `<log_file_parent>/plots` for single files.
    """
    log_files = _resolve_log_files(input_path)
    if output_dir is None:
        input_path_obj = Path(input_path)
        base_dir = input_path_obj if input_path_obj.is_dir() else input_path_obj.parent
        plot_dir = base_dir / "plots"
    else:
        plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_file in log_files:
        with np.load(log_file) as data:
            stem = log_file.stem
            trajectory_file = plot_dir / f"{stem}_trajectory.png"
            wrench_file = plot_dir / f"{stem}_wrench.png"
            _plot_trajectory(data, trajectory_file)
            _plot_wrench(data, wrench_file)
            logger.info("Saved plots for %s", log_file.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(analyze)
