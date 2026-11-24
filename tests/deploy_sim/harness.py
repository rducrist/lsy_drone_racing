"""Harness that runs the multi-drone deployment stack against the crazyflow_itl simulator.

The stack is the simulator, one estimator per drone, the race host and one client per drone, each
started as a subprocess in its own process group with its output written to a log file. Drone poses
are read from the /tf topic that the simulator publishes in place of the motion capture system.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import psutil
import rclpy
from rclpy.executors import SingleThreadedExecutor
from tf2_msgs.msg import TFMessage

from lsy_drone_racing.utils import load_config

REPO = Path(__file__).parents[2]
BASE_CONFIG = REPO / "config/multi_level2_sim.toml"
FAULT_CONTROLLER = Path(__file__).parent / "fault_controller.py"
SIM_BASE_PORT = 19850  # Hardcoded in the deploy envs as 19850 + rank


def wait_until(predicate: Callable[[], bool], timeout: float, interval: float = 0.05) -> bool:
    """Poll a predicate until it holds or the timeout expires and return whether it held."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def sim_ports_free(n_drones: int) -> bool:
    """Return whether the UDP ports of the simulated drones are free."""
    for rank in range(n_drones):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                sock.bind(("127.0.0.1", SIM_BASE_PORT + rank))
            except OSError:
                return False
    return True


def write_config(path: Path, controller: Path = FAULT_CONTROLLER) -> Path:
    """Write a copy of the multi-drone config that deploys to the simulator."""
    text = BASE_CONFIG.read_text()
    text, n = re.subn(r"(?m)^deploy_in_sim\s*=.*$", "deploy_in_sim = true", text)
    assert n == 1, "deploy_in_sim not found in the base config"
    text, n = re.subn(r'(?m)^file\s*=\s*"[^"]*"', f'file = "{controller}"', text)
    assert n >= 1, "No controller file entries found in the base config"
    path.write_text(text)
    return path


class Proc:
    """A subprocess in its own process group with its output in a log file."""

    def __init__(self, name: str, cmd: list[str], log_dir: Path, env: dict, cwd: Path = REPO):
        """Start the process."""
        self.name = name
        self.log_path = log_dir / f"{name}.log"
        self._log = open(self.log_path, "w")  # noqa: SIM115, closed in kill
        self.popen = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.pid = self.popen.pid  # Also the process group id, since it leads a new session

    def log(self) -> str:
        """Return the log so far."""
        return self.log_path.read_text(errors="replace")

    def mark(self) -> int:
        """Return the current log length to search only newer output with :meth:`wait_for`."""
        return len(self.log())

    def wait_for(self, pattern: str, timeout: float, since: int = 0) -> bool:
        """Wait until the log matches a regex after offset ``since``."""
        return wait_until(lambda: re.search(pattern, self.log()[since:]) is not None, timeout, 0.1)

    def alive(self) -> bool:
        """Return whether the leading process is still running."""
        return self.popen.poll() is None

    def group(self) -> list[psutil.Process]:
        """Return the live processes of the group, including orphaned children."""
        procs = []
        for p in psutil.process_iter(["status"]):
            try:
                if os.getpgid(p.pid) == self.pid and p.info["status"] != psutil.STATUS_ZOMBIE:
                    procs.append(p)
            except (ProcessLookupError, psutil.Error):
                pass
        return procs

    def signal(self, sig: int, group: bool = False):
        """Send a signal to the leading process, or to the whole group like a terminal does."""
        try:
            os.killpg(self.pid, sig) if group else os.kill(self.pid, sig)
        except ProcessLookupError:
            pass

    def kill(self):
        """Kill the whole group and close the log."""
        self.signal(signal.SIGKILL, group=True)
        try:
            self.popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._log.close()

    def tail(self, n: int = 40) -> str:
        """Return the last lines of the log."""
        return "\n".join(self.log().splitlines()[-n:])


class TFWatcher:
    """Track the positions of the frames on /tf of one ROS domain."""

    def __init__(self, domain_id: int):
        """Subscribe to /tf in its own ROS context and spin in a background thread."""
        self._context = rclpy.Context()
        rclpy.init(context=self._context, domain_id=domain_id)
        self.node = rclpy.create_node(f"deploy_sim_test_{uuid4().hex[:8]}", context=self._context)
        self._lock = threading.Lock()
        self._pos: dict[str, np.ndarray] = {}
        self._history: deque[tuple[float, str, np.ndarray]] = deque(maxlen=100_000)
        self.node.create_subscription(TFMessage, "/tf", self._callback, 100)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _callback(self, msg: TFMessage):
        with self._lock:
            for tf in msg.transforms:
                t = tf.transform.translation
                self._pos[tf.child_frame_id] = np.array([t.x, t.y, t.z])
                self._history.append(
                    (time.monotonic(), tf.child_frame_id, self._pos[tf.child_frame_id])
                )

    def trace(self, names: list[str], period: float = 0.5, last: float = 20.0) -> str:
        """Return the positions of frames over the last seconds, one line per period."""
        with self._lock:
            history = list(self._history)
        lines, t_next, latest = [], time.monotonic() - last, {}
        for t, name, pos in history:
            if name in names:
                latest[name] = pos
            if t >= t_next and len(latest) == len(names):
                cols = "  ".join(f"{n}={np.round(latest[n], 2)}" for n in names)
                lines.append(f"t={t - time.monotonic():6.1f}s  {cols}")
                t_next = t + period
        return "\n".join(lines)

    def pos(self, name: str) -> np.ndarray | None:
        """Return the latest position of a frame, or None if it was not seen yet."""
        with self._lock:
            return self._pos.get(name)

    def z(self, name: str) -> float:
        """Return the latest height of a frame, or NaN if it was not seen yet."""
        pos = self.pos(name)
        return float("nan") if pos is None else float(pos[2])

    def close(self):
        """Stop spinning, destroy the node and shut down the context."""
        self._executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        rclpy.shutdown(context=self._context)


class RaceStack:
    """The simulator, estimators, host and clients of one multi-drone race.

    Each stack runs on its own ROS domain. Processes of a previous stack that were killed leave
    stale Zenoh peers behind, and closing a session that knows them blocks for seconds.
    """

    def __init__(self, log_dir: Path, domain_id: int, drone_names: list[str]):
        """Prepare the config, environment and /tf watcher, without starting any process."""
        self.log_dir = log_dir
        self.watcher = TFWatcher(domain_id)
        self.drone_names = drone_names
        self.config = write_config(log_dir / "config.toml")
        self.env = os.environ | {
            "JAX_PLATFORMS": "cpu",
            "PYTHONUNBUFFERED": "1",
            "ROS_DOMAIN_ID": str(domain_id),
        }
        self.procs: list[Proc] = []
        self.host: Proc | None = None
        self.clients: dict[int, Proc] = {}
        self.estimators: dict[str, Proc] = {}
        drones = load_config(self.config).env.track.drones
        self.start_pos = [np.array(d["pos"]) for d in drones]

    def _start(self, name: str, cmd: list[str], env: dict | None = None, cwd: Path = REPO) -> Proc:
        proc = Proc(name, cmd, self.log_dir, env or self.env, cwd)
        self.procs.append(proc)
        return proc

    def start_sim(self, timeout: float = 240.0):
        """Start the simulator and wait until it publishes all drone poses."""
        cmd = [sys.executable, "scripts/deploy_in_sim.py", "--config", str(self.config)]
        sim = self._start("sim", cmd)
        ready = wait_until(
            lambda: all(self.watcher.pos(n) is not None for n in self.drone_names), timeout, 0.2
        )
        assert ready, f"Simulator did not publish {self.drone_names} on /tf\n{sim.tail()}"

    def start_estimators(self, estimator_type: str = "ukf", timeout: float = 180.0):
        """Start one estimator per drone and wait until all of them run."""
        for name in self.drone_names:
            cmd = [
                sys.executable,
                "-m",
                "drone_estimators.ros.node",
                "--drone_name",
                name,
                "--estimator_type",
                estimator_type,
            ]
            self.estimators[name] = self._start(f"estimator_{name}", cmd, cwd=self.log_dir)
        for proc in self.estimators.values():
            assert proc.wait_for(r"Started estimator", timeout), f"{proc.name}:\n{proc.tail()}"

    def start_host(self, timeout: float = 180.0) -> Proc:
        """Start the host and wait until it waits for the clients."""
        cmd = [sys.executable, "scripts/multi_deploy_host.py", "--config", str(self.config)]
        self.host = self._start("host", cmd)
        ready = self.host.wait_for(r"waiting for clients to register", timeout)
        assert ready, f"Host did not connect the drones\n{self.host.tail()}"
        return self.host

    def bring_up(self):
        """Start everything except the clients."""
        self.start_sim()
        self.start_estimators()
        self.start_host()

    def start_client(self, rank: int, fault: str = "none") -> Proc:
        """Start the client of a drone with an injected fault (see fault_controller.py)."""
        cmd = [
            sys.executable,
            "scripts/multi_deploy_client.py",
            "--config",
            str(self.config),
            "--drone_rank",
            str(rank),
        ]
        env = self.env | {"LSY_FAULT": fault}
        self.clients[rank] = self._start(f"client_{rank}", cmd, env=env)
        return self.clients[rank]

    def wait_registered(self, rank: int, timeout: float = 180.0) -> bool:
        """Wait until the host logs the registration of a drone's client."""
        return self.host.wait_for(rf"Client {self.drone_names[rank]} registered", timeout)

    def start_race(
        self, faults: dict[int, str] | None = None, timeout: float = 180.0, airborne: float = 0.2
    ):
        """Start all clients and wait until the race starts and the drones are airborne.

        Args:
            faults: Fault per rank (see fault_controller.py), ``none`` for unlisted ranks.
            timeout: Time to wait for the race start.
            airborne: Height that the drones of clients without a fault have to reach.
        """
        faults = faults or {}
        for rank in range(len(self.drone_names)):
            self.start_client(rank, faults.get(rank, "none"))
        assert self.host.wait_for(r"Race started", timeout), self.host.tail()
        healthy = [r for r in range(len(self.drone_names)) if faults.get(r, "none") == "none"]
        assert self.wait_heights(lambda z: z > airborne, 15.0, healthy), self.heights()

    def dist_to_start(self, rank: int) -> float:
        """Return the horizontal distance of a drone to its start position."""
        pos = self.watcher.pos(self.drone_names[rank])
        return float(np.linalg.norm(pos[:2] - self.start_pos[rank][:2]))

    def heights(self, ranks: list[int] | None = None) -> list[float]:
        """Return the current heights of the drones."""
        ranks = range(len(self.drone_names)) if ranks is None else ranks
        return [self.watcher.z(self.drone_names[r]) for r in ranks]

    def wait_heights(
        self, predicate: Callable[[float], bool], timeout: float, ranks: list[int] | None = None
    ) -> bool:
        """Wait until the height of every given drone satisfies a predicate."""
        return wait_until(lambda: all(predicate(z) for z in self.heights(ranks)), timeout)

    def trace(self) -> str:
        """Return the recent positions of all drones for failure messages."""
        return self.watcher.trace(self.drone_names)

    def close(self) -> str:
        """Kill all processes and return the tails of their logs and the drone trace."""
        for proc in reversed(self.procs):
            proc.kill()
        logs = [f"===== {p.name} ({p.log_path}) =====\n{p.tail()}" for p in self.procs]
        trace = self.trace()
        self.watcher.close()
        return "\n\n".join([*logs, f"===== drone positions =====\n{trace}"])
