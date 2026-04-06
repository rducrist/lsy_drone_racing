from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

import fire


EPISODE_RE = re.compile(r"Episode\s+(\d+):\s+(.*)")
DRONE_RE = re.compile(r"drone_(\d+)=(DNF|[0-9]*\.?[0-9]+)s?")


def _shell_command(repo_path: Path, command: str, ros_domain_id: int) -> str:
    quoted_repo = shlex.quote(str(repo_path.resolve()))
    return (
        f"cd {quoted_repo} && "
        f"export ROS_DOMAIN_ID={ros_domain_id} && "
        f"{command}"
    )


def _start_process(
    name: str,
    repo_path: Path,
    command: str,
    log_path: Path,
    ros_domain_id: int,
) -> tuple[subprocess.Popen[str], object]:
    if not repo_path.exists():
        raise FileNotFoundError(f"{name} repo does not exist: {repo_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w")
    process = subprocess.Popen(
        ["bash", "-lc", _shell_command(repo_path, command, ros_domain_id)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    print(f"Started {name}: pid={process.pid}, repo={repo_path}")
    return process, log_handle


def _terminate_process(process: subprocess.Popen[str], name: str, grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.1)
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        print(f"Stopped {name}: pid={process.pid}")


def _parse_sim_log(log_path: Path) -> tuple[list[list[float | None]], list[bool]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Simulator log not found: {log_path}")

    episode_times: dict[int, list[float | None]] = {}
    for line in log_path.read_text().splitlines():
        match = EPISODE_RE.search(line)
        if not match:
            continue
        episode_index = int(match.group(1))
        payload = match.group(2)
        drone_times: dict[int, float | None] = {}
        for drone_match in DRONE_RE.finditer(payload):
            drone_id = int(drone_match.group(1))
            raw_time = drone_match.group(2)
            drone_times[drone_id] = None if raw_time == "DNF" else float(raw_time)

        if drone_times:
            max_drone_id = max(drone_times)
            episode_times[episode_index] = [
                drone_times.get(drone_id) for drone_id in range(max_drone_id + 1)
            ]

    if not episode_times:
        raise RuntimeError(f"Could not parse any episode results from simulator log: {log_path}")

    ordered_episode_ids = sorted(episode_times)
    ordered_times = [episode_times[idx] for idx in ordered_episode_ids]
    episode_finished = [all(t is not None for t in episode) for episode in ordered_times]
    return ordered_times, episode_finished


def _summarize_pair_result(
    episode_times: list[list[float | None]],
    episode_finished: list[bool],
    team_a: str,
    team_b: str,
    controller_a_repo: Path,
    controller_b_repo: Path,
    sim_repo: Path,
) -> dict:
    average_times: list[float | None] = []
    for drone_id in range(2):
        finished_times = [
            episode[drone_id]
            for episode in episode_times
            if drone_id < len(episode) and episode[drone_id] is not None
        ]
        average_times.append(sum(finished_times) / len(finished_times) if finished_times else None)

    if average_times[0] is None and average_times[1] is None:
        winner: str | None = None
    elif average_times[1] is None or (
        average_times[0] is not None and average_times[0] < average_times[1]
    ):
        winner = team_a
    elif average_times[0] is None or average_times[1] < average_times[0]:
        winner = team_b
    else:
        winner = "tie"

    return {
        "team_a": team_a,
        "team_b": team_b,
        "sim_repo": str(sim_repo.resolve()),
        "controller_a_repo": str(controller_a_repo.resolve()),
        "controller_b_repo": str(controller_b_repo.resolve()),
        "episode_times": episode_times,
        "episode_finished": episode_finished,
        "average_time_a": average_times[0],
        "average_time_b": average_times[1],
        "winner": winner,
    }


def run(
    sim_repo: str,
    controller_a_repo: str,
    controller_b_repo: str,
    output_file: str,
    team_a: str = "team_a",
    team_b: str = "team_b",
    logs_dir: str | None = None,
    sim_config: str = "multi_level0.toml",
    n_runs: int = 1,
    action_timeout: float = 10.0,
    startup_delay: float = 3.0,
    timeout: float = 300.0,
    controller_a: str = "attitude_controller.py",
    controller_b: str = "attitude_controller.py",
    ros_domain_id: int = 0,
):
    """Run one pair evaluation by orchestrating existing scripts headlessly."""
    sim_repo_path = Path(sim_repo)
    ctrl_a_repo_path = Path(controller_a_repo)
    ctrl_b_repo_path = Path(controller_b_repo)
    output_path = Path(output_file)
    logs_path = Path(logs_dir) if logs_dir else output_path.parent / f"{output_path.stem}_logs"

    logs_path.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sim_log_path = logs_path / "sim.log"
    ctrl_a_log_path = logs_path / "controller_a.log"
    ctrl_b_log_path = logs_path / "controller_b.log"

    sim_command = (
        "pixi run -e deploy python scripts/multi_sim_node.py "
        f"--config={shlex.quote(sim_config)} "
        f"--n-runs={n_runs} "
        "--gui=False "
        f"--action-timeout={action_timeout}"
    )
    ctrl_a_command = (
        "pixi run -e deploy python scripts/controller_node.py "
        f"--drone-id=0 --config={shlex.quote(sim_config)} "
        f"--controller={shlex.quote(controller_a)}"
    )
    ctrl_b_command = (
        "pixi run -e deploy python scripts/controller_node.py "
        f"--drone-id=1 --config={shlex.quote(sim_config)} "
        f"--controller={shlex.quote(controller_b)}"
    )

    sim_process = ctrl_a_process = ctrl_b_process = None
    sim_log = ctrl_a_log = ctrl_b_log = None
    try:
        sim_process, sim_log = _start_process(
            "simulator", sim_repo_path, sim_command, sim_log_path, ros_domain_id
        )
        time.sleep(startup_delay)
        ctrl_a_process, ctrl_a_log = _start_process(
            "controller_a", ctrl_a_repo_path, ctrl_a_command, ctrl_a_log_path, ros_domain_id
        )
        ctrl_b_process, ctrl_b_log = _start_process(
            "controller_b", ctrl_b_repo_path, ctrl_b_command, ctrl_b_log_path, ros_domain_id
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sim_code = sim_process.poll()
            ctrl_a_code = ctrl_a_process.poll()
            ctrl_b_code = ctrl_b_process.poll()

            if sim_code is not None and ctrl_a_code is not None and ctrl_b_code is not None:
                break
            if sim_code not in (None, 0):
                raise RuntimeError(f"Simulator failed with exit code {sim_code}")
            if ctrl_a_code not in (None, 0):
                raise RuntimeError(f"Controller A failed with exit code {ctrl_a_code}")
            if ctrl_b_code not in (None, 0):
                raise RuntimeError(f"Controller B failed with exit code {ctrl_b_code}")
            time.sleep(0.5)
        else:
            raise TimeoutError(f"Pair evaluation exceeded timeout of {timeout} seconds")

        if sim_process.returncode != 0:
            raise RuntimeError(f"Simulator failed with exit code {sim_process.returncode}")
        if ctrl_a_process.returncode != 0:
            raise RuntimeError(f"Controller A failed with exit code {ctrl_a_process.returncode}")
        if ctrl_b_process.returncode != 0:
            raise RuntimeError(f"Controller B failed with exit code {ctrl_b_process.returncode}")

        episode_times, episode_finished = _parse_sim_log(sim_log_path)
        pair_result = _summarize_pair_result(
            episode_times=episode_times,
            episode_finished=episode_finished,
            team_a=team_a,
            team_b=team_b,
            controller_a_repo=ctrl_a_repo_path,
            controller_b_repo=ctrl_b_repo_path,
            sim_repo=sim_repo_path,
        )
        pair_result["logs_dir"] = str(logs_path.resolve())
        output_path.write_text(json.dumps(pair_result, indent=2))
        print(f"Wrote pair result to {output_path.resolve()}")
    finally:
        for process, name in [
            (ctrl_a_process, "controller_a"),
            (ctrl_b_process, "controller_b"),
            (sim_process, "simulator"),
        ]:
            if process is not None:
                _terminate_process(process, name)
        for handle in [sim_log, ctrl_a_log, ctrl_b_log]:
            if handle is not None:
                handle.close()


if __name__ == "__main__":
    fire.Fire(run)
