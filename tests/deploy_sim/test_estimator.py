"""Behavior of the multi-drone deployment when a drone's state estimate stops updating."""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.deploy_sim.harness import wait_until

if TYPE_CHECKING:
    from tests.deploy_sim.harness import RaceStack

pytestmark = pytest.mark.deploy_sim

LANDED = 0.1  # Height in m below which a drone counts as down


# Not strict: on a frozen estimate the drone sometimes falls inside the safety box by chance,
# which this test cannot tell apart from being stopped
@pytest.mark.xfail(
    strict=False,
    reason="Nothing checks the age of the estimates, the client flies on a frozen state",
)
def test_estimator_loss_stops_drone(stack: RaceStack):
    """A drone whose estimate freezes is stopped instead of flying on a stale state."""
    stack.bring_up()
    stack.start_race()

    stack.estimators[stack.drone_names[0]].signal(signal.SIGKILL, group=True)

    # Track the true pose from the simulator while waiting for the drone to come down
    low = np.array([-2.5, -1.5, -1e-2])  # Safety limits of multi_level2.toml
    high = np.array([2.5, 1.5, 2.0])
    left_bounds = []

    def down() -> bool:
        pos = stack.watcher.pos(stack.drone_names[0])
        if np.any(pos < low) or np.any(pos > high):
            left_bounds.append(pos.copy())
        return pos[2] < LANDED

    assert wait_until(down, 3.0), f"Drone 0 still flies on a frozen estimate: {stack.heights()}"
    assert not left_bounds, f"Drone 0 left the safety box at {left_bounds[0]}"
