"""Multi-drone attitude controller for deploy-in-sim tests that can inject client faults.

The ``LSY_FAULT`` environment variable selects the fault, timed from the first control step:

* ``none``: Follow the trajectory of :mod:`lsy_drone_racing.control.attitude_controller_multi`.
* ``raise@T``: Raise an exception after ``T`` seconds.
* ``hang@T``: Block forever after ``T`` seconds.
* ``slow=DT``: Sleep ``DT`` seconds in every control step.
"""

from __future__ import annotations

import os
import re
import time
from typing import TYPE_CHECKING

from lsy_drone_racing.control.attitude_controller_multi import AttitudeController

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class FaultController(AttitudeController):
    """Attitude controller with an injected fault."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Parse the fault and initialize the attitude controller."""
        super().__init__(obs, info, config)
        fault = os.environ.get("LSY_FAULT", "none")
        match = re.fullmatch(r"(none|raise|hang|slow)(?:[@=]([0-9.]+))?", fault)
        assert match, f"Invalid LSY_FAULT {fault!r}"
        self._fault, self._arg = match[1], float(match[2] or 0.0)
        self._t0: float | None = None

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Inject the fault and compute the control of the attitude controller."""
        now = time.monotonic()
        self._t0 = now if self._t0 is None else self._t0
        elapsed = now - self._t0
        if self._fault == "raise" and elapsed >= self._arg:
            raise RuntimeError("Injected controller fault")
        if self._fault == "hang" and elapsed >= self._arg:
            while True:
                time.sleep(1.0)
        if self._fault == "slow":
            time.sleep(self._arg)
        return super().compute_control(obs, info)
