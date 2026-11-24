"""Behavior of the multi-drone deployment when a client or the host fails mid-race."""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

import pytest

from tests.deploy_sim.harness import wait_until

if TYPE_CHECKING:
    from tests.deploy_sim.harness import RaceStack

pytestmark = pytest.mark.deploy_sim

LANDED = 0.1  # Height in m below which a drone counts as down


def test_client_killed_mid_race_stops_its_drone(stack: RaceStack):
    """A worker whose client dies stops its drone, and the other drones keep racing."""
    stack.bring_up()
    stack.start_race()
    mark = stack.host.mark()

    stack.clients[0].signal(signal.SIGKILL, group=True)

    # The worker detects missing actions after 10 control periods (0.2 s at 50 Hz)
    assert stack.host.wait_for(r"\[Drone 0\] ERROR: No command received", 2.0, since=mark), (
        stack.host.tail()
    )
    assert stack.wait_heights(lambda z: z < LANDED, 5.0, ranks=[0]), stack.heights()
    assert stack.heights([1])[0] > 0.2, f"Drone 1 stopped racing: {stack.heights()}"


def test_host_interrupt_stops_all_drones(stack: RaceStack):
    """Ctrl+C on the host stops every drone and ends every host process."""
    stack.bring_up()
    stack.start_race()

    stack.host.signal(signal.SIGINT, group=True)  # Like Ctrl+C in a terminal

    assert stack.wait_heights(lambda z: z < LANDED, 5.0), f"Drones still up: {stack.heights()}"
    assert wait_until(lambda: not stack.host.group(), 30.0), (
        f"Host processes left: {stack.host.group()}\n{stack.host.tail()}"
    )
    assert "Host shutdown complete" in stack.host.log()


@pytest.mark.xfail(
    strict=True, reason="Workers do not watch the host process and keep forwarding client actions"
)
def test_host_crash_stops_all_drones(stack: RaceStack):
    """Drones stop if the host process dies without cleaning up, e.g. on SIGKILL."""
    stack.bring_up()
    stack.start_race()

    stack.host.signal(signal.SIGKILL)  # Only the host, its workers become orphans

    assert stack.wait_heights(lambda z: z < LANDED, 5.0), f"Drones still up: {stack.heights()}"


def test_race_finish_prints_leaderboard_and_exits(stack: RaceStack):
    """After a full race the host prints the leaderboard and all host processes exit."""
    stack.bring_up()
    stack.start_race()

    assert stack.host.wait_for(r"Leaderboard:", 120.0), stack.host.tail()
    log = stack.host.log()
    for name in stack.drone_names:
        assert f". {name} " in log, f"{name} missing from the leaderboard\n{stack.host.tail()}"
    assert wait_until(lambda: not stack.host.group(), 30.0), (
        f"Host processes left: {stack.host.group()}\n{stack.host.tail()}"
    )
