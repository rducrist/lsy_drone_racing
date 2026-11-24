"""Behavior of the multi-drone deployment when a client's controller misbehaves."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from lsy_drone_racing.envs.real_race_host_env import CrazyflieWorker
from tests.deploy_sim.harness import wait_until

if TYPE_CHECKING:
    from tests.deploy_sim.harness import RaceStack

pytestmark = pytest.mark.deploy_sim

LANDED = 0.1  # Height in m below which a drone counts as down
STALE = r"\[Drone 0\] ERROR: No command received"  # Worker of rank 0 detected missing actions


def test_client_without_first_action_is_stopped(stack: RaceStack):
    """A worker whose client never sends an action gives up instead of waiting forever."""
    stack.bring_up()
    stack.start_race(faults={0: "hang@0"})  # Client 0 blocks in its first control step

    timeout = CrazyflieWorker.FIRST_ACTION_TIMEOUT + 3.0
    assert stack.host.wait_for(STALE, timeout), f"Worker 0 still waits\n{stack.host.tail()}"


def test_controller_crash_returns_drone_home(stack: RaceStack):
    """A client whose controller raises deregisters, and its drone flies back to its start."""
    stack.bring_up()
    stack.start_race(faults={0: "raise@3"})

    # The client prints the traceback only on exit, after its shutdown, so watch the host instead
    assert stack.host.wait_for(rf"Client {stack.drone_names[0]} stopped", 20.0), stack.host.tail()
    # Return to start: brake (3 s), fly at the return height (5 s), land (3 s)
    assert stack.host.wait_for(r"\[Drone 0\] INFO: Returning to start", 3.0), stack.host.tail()
    assert stack.wait_heights(lambda z: z > 1.5, 10.0, ranks=[0]), stack.heights()
    home = wait_until(lambda: stack.dist_to_start(0) < 0.2 and stack.heights([0])[0] < 0.15, 20.0)
    assert home, f"Drone 0 not home: {stack.dist_to_start(0):.2f} m away, {stack.heights()}"
    assert not re.search(STALE, stack.host.log()), "Crash taken for missing actions"


def test_slow_controller_within_timeout_keeps_flying(stack: RaceStack):
    """A client that overruns its period but stays below the worker timeout keeps control."""
    stack.bring_up()
    stack.start_race(faults={0: "slow=0.03"})  # 30 ms per step at 50 Hz, the timeout is 200 ms
    assert stack.wait_heights(lambda z: z > 0.2, 15.0, ranks=[0]), stack.heights()

    assert not stack.host.wait_for(STALE, 5.0), f"Worker 0 timed out\n{stack.host.tail()}"
    assert stack.heights([0])[0] > 0.2, f"Drone 0 came down: {stack.heights()}"


def test_slow_controller_beyond_timeout_is_stopped(stack: RaceStack):
    """A client that sends actions slower than the worker timeout loses its drone."""
    stack.bring_up()
    stack.start_race(faults={0: "slow=0.3"})  # One action every 300 ms, the timeout is 200 ms

    assert stack.host.wait_for(STALE, 5.0), f"Worker 0 kept waiting\n{stack.host.tail()}"
    assert stack.wait_heights(lambda z: z < LANDED, 5.0, ranks=[0]), stack.heights()
