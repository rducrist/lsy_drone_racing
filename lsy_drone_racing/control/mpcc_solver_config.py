"""This class is meant to contain all the hyperparameters used for MPCC and OCP solver. All the magic numbers go in here."""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MPCCSolverConfig:
    """This dataclass contains mpcc and solver hyperparameters."""
    
    N: int = 40
    T_horizon: float = 0.8

    # PMM planner parameters for waypoint generation
    distance_before : float = 0.3
    distance_after : float = 0.2
    end_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sensor_range : float = 0.6

    # MPCC path discretization
    model_traj_length: float = 10
    delta_theta: float = 0.05

    # MPCC progress params
    q_lag: float = 250.0              # Lag error weight at gates
    q_lag_peak: float = 0
    q_contour: float = 350.0          # Contour error weight at gates
    q_contour_peak: float = 0
    q_attitude: float = 1.0

    # Control smoothness
    r_thrust: float = 0.1                 # Thrust rate penalty
    r_roll: float = 0.1                    # Roll rate penalty
    r_pitch: float = 0.1                   # Pitch rate penalty
    r_yaw: float = 0.1
    
    mu_speed: float = 5.0 


    @property
    def dt(self) -> float:
        """Sets the discretisation time for the pmm planner."""
        return self.T_horizon / self.N

    @property
    def M(self) -> int:
        """Sets the length for the parametrized path. Used in solver file."""
        return self.theta_grid.shape[0]

    @property
    def theta_grid(self) -> np.ndarray:
        """Creates a uniformly discretized grid along the parametrized path."""
        return np.arange(0.0, self.model_traj_length, self.delta_theta)
    