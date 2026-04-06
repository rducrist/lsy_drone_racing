from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import fire


def _terminal_launcher() -> list[str]:
    """Return a terminal command prefix available on the current Linux desktop."""
    candidates = [
        ["x-terminal-emulator", "-e"],
        ["gnome-terminal", "--"],
        ["konsole", "-e"],
        ["xfce4-terminal", "-e"],
        ["xterm", "-e"],
    ]
    for candidate in candidates:
        if shutil.which(candidate[0]):
            return candidate
    raise RuntimeError(
        "No supported terminal emulator found. Install one of: "
        "x-terminal-emulator, gnome-terminal, konsole, xfce4-terminal, xterm."
    )


def _open_terminal(repo_path: Path, title: str, pixi_command: str) -> subprocess.Popen[bytes]:
    """Open a new terminal window and run a Pixi command inside the given repo."""
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    launcher = _terminal_launcher()
    quoted_repo = shlex.quote(str(repo_path.resolve()))
    command = (
        f"printf '\\033]0;{title}\\007'; "
        f"cd {quoted_repo} && "
        f"pixi run -e deploy {pixi_command}; "
        "exec bash"
    )
    return subprocess.Popen([*launcher, "bash", "-lc", command])


def launch(
    sim_repo: str = "/home/radu/Uni/hiwi_lsy/lsy_drone_racing/",
    controller_a_repo: str = "/home/radu/Uni/hiwi_lsy/lsy_drone_racing/",
    controller_b_repo: str = "/home/radu/Uni/hiwi_lsy/lsy_drone_racing/",
    sim_task: str = "multi-sim-node",
    controller_task: str = "controller-node",
    sim_args: str = "--config=multi_level0.toml --n-runs=1",
    controller_a_args: str = "--drone-id=0",
    controller_b_args: str = "--drone-id=1",
):
    r"""Launch a minimal local competition setup in fresh terminals.

    Example:
        pixi run manual-competition-example \\
          --sim-repo=/abs/path/base_repo \\
          --controller-a-repo=/abs/path/fork_a \\
          --controller-b-repo=/abs/path/fork_b \\
          --sim-task=multi-sim-node \\
          --sim-args="--config=multi_level0.toml --n-runs=1" \\
          --controller-task=controller-node

    Notes:
        - Each process runs in its own terminal and therefore in its own Pixi environment.
        - All commands are executed with `pixi run -e deploy ...`.
        - `sim_args`, `controller_a_args`, and `controller_b_args` are appended after the task name.
        - This is intended for a local Linux desktop session, not for GitHub Actions.
    """
    sim_process = _open_terminal(Path(sim_repo), "multi-sim", f"{sim_task} {sim_args}".strip())
    ctrl_a_process = _open_terminal(
        Path(controller_a_repo),
        "controller-0",
        f"{controller_task} {controller_a_args}".strip(),
    )
    ctrl_b_process = _open_terminal(
        Path(controller_b_repo),
        "controller-1",
        f"{controller_task} {controller_b_args}".strip(),
    )

    print("Opened terminals:")
    print(f"  simulator:    pid={sim_process.pid} repo={Path(sim_repo).resolve()}")
    print(f"  controller 0: pid={ctrl_a_process.pid} repo={Path(controller_a_repo).resolve()}")
    print(f"  controller 1: pid={ctrl_b_process.pid} repo={Path(controller_b_repo).resolve()}")


if __name__ == "__main__":
    fire.Fire(launch)
