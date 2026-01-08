from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MPCCSolverConfig:
    # MPC horizon
    N: int = 40
    T_horizon: float = 1.0

    # MPCC path discretization
    model_traj_length: float = 12.0
    delta_theta: float = 0.05

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N

    @property
    def M(self) -> int:
        return int(self.model_traj_length / self.delta_theta)

    @property
    def theta_grid(self) -> np.ndarray:
        return np.arange(0.0, self.model_traj_length, self.delta_theta)