"""Wrapper for allowing the use of example controllers with batched observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from lsy_drone_racing.control import Controller


class MultiAgentControllerWrapper(Controller):
    """Wrap a single agent controller behind a multi-agent interface."""

    controller_cls: type[Controller] | None = None

    def __init__(self, obs: dict[str, Any], info: dict, config: dict):
        """Initializes the controller with the passed controller class and rank."""
        assert self.controller_cls is not None, "controller_cls must be set by subclasses"
        self.rank = int((info or {}).get("rank", 0))
        self.controller = self.controller_cls(self._slice_obs(obs, self.rank), info, config)

    @staticmethod
    def _slice_obs(obs: dict[str, Any], rank: int) -> dict[str, Any]:
        """Select observations corresponding to a single drone."""
        return {k: np.asarray(v)[rank] for k, v in obs.items()}


    def compute_control(self, obs: dict[str, Any], info: dict | None = None) -> np.ndarray:
        """Calls compute control of the controller class with the sliced observations."""
        return self.controller.compute_control(self._slice_obs(obs, self.rank), info)

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Calls step callback of the controller with the sliced observations."""
        return self.controller.step_callback(
            action,
            self._slice_obs(obs, self.rank),
            reward, 
            terminated,
            truncated, 
            info,
        )

    def episode_callback(self) -> None:
        """Forward episode callback to wrapped controller."""
        return self.controller.episode_callback()

    def episode_reset(self) -> None:
        """Forward episode reset to wrapped controller."""
        return self.controller.episode_reset()


def wrap_controller(controller_cls: type[Controller]) -> type[MultiAgentControllerWrapper]:
    """Create multi-agent wrapper class for a single-agent controller class."""

    class WrappedMultiAgentController(MultiAgentControllerWrapper):
        """Dynamic wrapper around a single-agent controller class."""

        pass

    WrappedMultiAgentController.controller_cls = controller_cls
    WrappedMultiAgentController.__name__ = f"Multi{controller_cls.__name__}"
    WrappedMultiAgentController.__qualname__ = WrappedMultiAgentController.__name__
    return WrappedMultiAgentController
