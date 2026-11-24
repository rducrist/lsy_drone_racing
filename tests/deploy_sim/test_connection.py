"""Race start conditions of the multi-drone deployment."""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.deploy_sim.harness import RaceStack

pytestmark = pytest.mark.deploy_sim


def test_race_waits_for_all_clients(stack: RaceStack):
    """The race starts only once every client has registered."""
    stack.bring_up()
    stack.start_client(0)
    assert stack.wait_registered(0), stack.host.tail()

    assert not stack.host.wait_for(r"Race started", 10.0), "Race started with one client"
    assert max(stack.heights()) < 0.05, f"A drone took off before the race: {stack.heights()}"

    stack.start_client(1)
    assert stack.host.wait_for(r"Race started", 180.0), stack.host.tail()
    took_off = stack.wait_heights(lambda z: z > 0.2, 15.0)
    assert took_off, f"Drones did not take off: {stack.heights()}"


@pytest.mark.xfail(
    strict=True,
    reason="The client skips the deregister on Ctrl+C, and the host keeps counting it registered",
)
def test_client_interrupted_before_start_blocks_race(stack: RaceStack):
    """A client that is interrupted before the race starts must keep the race from starting."""
    stack.bring_up()
    client = stack.start_client(0)
    assert stack.wait_registered(0), stack.host.tail()
    client.signal(signal.SIGINT, group=True)
    client.popen.wait(timeout=30)

    stack.start_client(1)
    assert stack.wait_registered(1), stack.host.tail()
    assert not stack.host.wait_for(r"Race started", 15.0), "Race started without client 0"
