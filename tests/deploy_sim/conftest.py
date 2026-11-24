"""Fixtures for the deploy-in-sim tests.

These tests need the deploy environment and a free simulator port, and take about a minute each:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v tests/deploy_sim

ROS' launch_testing pytest plugins break pytest in the deploy environment, hence the variable.
"""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, Iterator

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("crazyflow_itl")

from lsy_drone_racing.utils import load_config  # noqa: E402
from tests.deploy_sim.harness import BASE_CONFIG, RaceStack, sim_ports_free  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

DRONE_NAMES = [f"cf{d['id']}" for d in load_config(BASE_CONFIG).deploy.drones]
# One ROS domain per test keeps the tests away from each other, other ROS processes on this
# machine and the network
DOMAIN_IDS = itertools.count(random.randint(100, 150))


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[RaceStack]:
    """Provide a race stack whose processes are all killed after the test."""
    if not sim_ports_free(len(DRONE_NAMES)):
        pytest.skip("A simulator already uses the drone UDP ports, stop it to run these tests")
    stack = RaceStack(tmp_path, next(DOMAIN_IDS), DRONE_NAMES)
    yield stack
    print(stack.close())  # Shown by pytest for failed tests
