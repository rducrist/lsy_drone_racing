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
    # Tracking cost 
    q_lag: float = 250.0              # Lag error weight at gates
    q_contour: float = 350.0          # Contour error weight at gates
    q_attitude: float = 1.0 

    # Control smoothness
    r_thrust: float = 0.2                  # Thrust rate penalty
    r_roll: float = 0.3                    # Roll rate penalty
    r_pitch: float = 0.3                   # Pitch rate penalty
    r_yaw: float = 0.50   
    
    mu_speed: float = 1.0 


    @property
    def dt(self) -> float:
        return self.T_horizon / self.N

    @property
    def M(self) -> int:
        return self.theta_grid.shape[0]

    @property
    def theta_grid(self) -> np.ndarray:
        return np.arange(0.0, self.model_traj_length, self.delta_theta)
    