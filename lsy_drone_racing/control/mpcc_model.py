"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
from casadi import MX, cos, sin, vertcat
import drone_models.symbols as symbols
from drone_models.utils import rotation

if TYPE_CHECKING:
    from array_api_typing import Array


def symbolic_dynamics_euler_mpcc_so_rpy_rotor(
    model_rotor_vel: bool = False,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """The fitted linear, second order rpy dynamics with thrust dynamics.

    For info on the args, see above.

    This function returns the actual model, as defined in the paper, for direct use.
    """
    """Build the quadrotor dynamics model."""
    
    # Rate model parameters (from system identification)
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    
    # State variables
    px = MX.sym("px")
    py = MX.sym("py")
    pz = MX.sym("pz")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    vz = MX.sym("vz")
    roll = MX.sym("roll")
    pitch = MX.sym("pitch")
    yaw = MX.sym("yaw")
    f_collective = MX.sym("f_collective")
    f_cmd = MX.sym("f_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")
    theta = MX.sym("theta")  # Progress along path
    
    # Input variables
    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")
    v_theta_cmd = MX.sym("v_theta_cmd")  # Progress speed
    
    # State and input vectors
    states = vertcat(
        px, py, pz,
        vx, vy, vz,
        roll, pitch, yaw,
        f_collective, f_cmd,
        r_cmd, p_cmd, y_cmd,
        theta
    )
    inputs = vertcat(
        df_cmd, dr_cmd, dp_cmd, dy_cmd,
        v_theta_cmd
    )
    
    # Dynamics equations
    thrust = f_collective
    inv_mass = 1.0 / mass
    
    # Acceleration from thrust
    ax = inv_mass * thrust * (
        cos(roll) * sin(pitch) * cos(yaw)
        + sin(roll) * sin(yaw)
    )
    ay = inv_mass * thrust * (
        cos(roll) * sin(pitch) * sin(yaw)
        - sin(roll) * cos(yaw)
    )
    az = inv_mass * thrust * cos(roll) * cos(pitch) + gravity_vec[2]
    
    # Continuous dynamics
    f_dyn = vertcat(
        vx,
        vy,
        vz,
        ax,
        ay,
        az,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
        v_theta_cmd
    )

    return f_dyn, states, inputs
