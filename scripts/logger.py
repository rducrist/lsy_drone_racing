from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

if TYPE_CHECKING:
    from lsy_drone_racing.envs.multi_drone_race import MultiDroneRacingEnv



logger = logging.getLogger(__name__)


@dataclass
class EpisodeLogger:
    """Collect per-step simulation data and write it to disk."""

    log_dir: Path
    controller_names: list[str]
    enabled: bool = True
    _runs: list[dict[str, np.ndarray | list[str] | int | float | str]] = field(default_factory=list)

    def __post_init__(self):
        """Create the output directory only when logging is enabled."""
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_run(
        self,
        run_idx: int,
        base_freq: int,
        controller_freqs: np.ndarray,
        samples: list[dict[str, np.ndarray | float | bool]],
    ):
        """Store a completed run and flush it to disk."""
        if not self.enabled:
            return
        if not samples:
            logger.warning("Run %d did not produce any samples, skipping log write.", run_idx)
            return

        times = np.array([sample["t"] for sample in samples], dtype=np.float32)
        actions = np.stack([sample["action"] for sample in samples]).astype(np.float32)
        action_updated = np.stack([sample["action_updated"] for sample in samples]).astype(bool)
        reward = np.array([sample["reward"] for sample in samples], dtype=np.float32)
        terminated = np.array([sample["terminated"] for sample in samples], dtype=bool)
        truncated = np.array([sample["truncated"] for sample in samples], dtype=bool)
        disabled = np.stack([sample["disabled"] for sample in samples]).astype(bool)
        target_gate = np.stack([sample["target_gate"] for sample in samples]).astype(np.int32)
        pos = np.stack([sample["pos"] for sample in samples]).astype(np.float32)
        quat = np.stack([sample["quat"] for sample in samples]).astype(np.float32)
        vel = np.stack([sample["vel"] for sample in samples]).astype(np.float32)
        ang_vel = np.stack([sample["ang_vel"] for sample in samples]).astype(np.float32)
        force = np.stack([sample["force"] for sample in samples]).astype(np.float32)
        torque = np.stack([sample["torque"] for sample in samples]).astype(np.float32)

        file = self.log_dir / f"multi_sim_run_{run_idx:03d}.npz"
        np.savez_compressed(
            file,
            t=times,
            action=actions,
            action_updated=action_updated,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            disabled=disabled,
            target_gate=target_gate,
            pos=pos,
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
            force=force,
            torque=torque,
            controller_names=np.array(self.controller_names),
            base_freq=np.int32(base_freq),
            controller_freqs=np.asarray(controller_freqs, dtype=np.int32),
        )
        logger.info("Saved simulation log to %s", file)


    def _to_numpy(self, value) -> np.ndarray:
        """Convert JAX / numpy values to numpy arrays without changing shape."""
        return np.asarray(value)


    def _extract_torque(self, states, force: np.ndarray) -> np.ndarray:
        """Read torque from the simulator state if available, otherwise return zeros."""
        for attr in ("torque", "tau", "moments", "moment"):
            if hasattr(states, attr):
                return self._to_numpy(getattr(states, attr))
        logger.warning("Could not find a torque field in sim states. Logging zeros instead.")
        return np.zeros_like(force, dtype=np.float32)


    def _snapshot_step(
        self,
        env: MultiDroneRacingEnv,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        actions: np.ndarray,
        action_updated: np.ndarray,
        t: float,
    ) -> dict[str, np.ndarray | float | bool]:
        """Capture a single simulation sample for all drones."""
        data = env.unwrapped.data
        states = data.sim_data.states
        force = self._to_numpy(states.force)
        torque = self._extract_torque(states, force)
        return {
            "t": t,
            "action": actions.copy(),
            "action_updated": action_updated.copy(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "disabled": self._to_numpy(data.disabled_drones[0]),
            "target_gate": self._to_numpy(obs["target_gate"]),
            "pos": self._to_numpy(obs["pos"]),
            "quat": self._to_numpy(obs["quat"]),
            "vel": self._to_numpy(obs["vel"]),
            "ang_vel": self._to_numpy(obs["ang_vel"]),
            "force": force[0].copy(),
            "torque": torque[0].copy(),
        }