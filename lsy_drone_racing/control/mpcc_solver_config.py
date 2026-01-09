from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MPCCSolverConfig:
    # MPC horizon
    N: int = 40
    T_horizon: float = 1.0

    # MPCC path discretization
    model_traj_length: float = 9.14
    delta_theta: float = 0.05

    # MPCC progress params
    qc : int = 300.0
    ql : int = 250.0
    mu : int = 3.0

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N

    @property
    def M(self) -> int:
        return self.theta_grid.shape[0]

    @property
    def theta_grid(self) -> np.ndarray:
        return np.arange(0.0, self.model_traj_length, self.delta_theta)
    
    @property
    def get_progress_params(self) -> np.ndarray:
        """Sdfs."""
        return np.array([self.qc, self.ql, self.mu])