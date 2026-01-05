"""This class implements a logger for the controller.

We store here only the value than we want to plot later on. No more print statements!
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from numpy.typing import NDArray



class MPCLogger:
    def __init__(self):
        self.solver_times: List[float] = []
        self.costs: List[float] = []
        self.predicted_trajs: List[NDArray] = []
        self.current_prediction : NDArray =[]
        self.states: List[NDArray] = []
        self.controls: List[NDArray] = []
        self.gate_inner_ring: NDArray = []
        self.gate_outer_ring: NDArray = []

    def log_step(
        self,
        solver_time: float | None = None,
        cost: float | None = None,
        predictions: NDArray | None = None,
        state: NDArray | None = None,
        control: NDArray | None = None,
        timestamp: float | None = None,
        gate_inner_ring: NDArray | None = None,
        gate_outer_ring: NDArray | None = None
    ) -> None:
        """Defines what has to logged each step."""
        if solver_time is not None:
            self.solver_times.append(solver_time)
        if cost is not None:
            self.costs.append(cost)
        if predictions is not None:
            self.predicted_trajs.append(predictions.copy())
            self.current_prediction = predictions
        if state is not None:
            self.states.append(state.copy())
        if control is not None:
            self.controls.append(control.copy())
        if timestamp is not None:
            self.timestamps.append(timestamp)
        if gate_inner_ring is not None:
            self.gate_inner_ring = gate_inner_ring
        if gate_outer_ring is not None:
            self.gate_outer_ring = gate_outer_ring    

        