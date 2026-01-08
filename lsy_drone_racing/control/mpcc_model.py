"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
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

    # MPCC progress variables 
    theta = cs.MX.sym("theta")
    d_theta = cs.MX.sym("d_theta")
    cmd_dd_theta = cs.MX.sym("cmd_dd_theta")

    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.rpy, symbols.vel, symbols.drpy, theta, d_theta)
    if model_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    U = cs.vertcat(symbols.cmd_rpyt, cmd_dd_theta)
    cmd_rpy = U[:3]
    cmd_thrust = U[3]
    cmd_dd_theta = U[4]
    rot = rotation.cs_rpy2matrix(symbols.rpy)

    # Defining the dynamics function
    # Note that we are abusing the rotor_vel state as the thrust
    if model_rotor_vel:
        rotor_vel_dot = 1 / thrust_time_coef * (cmd_thrust - symbols.rotor_vel)
        forces_motor = symbols.rotor_vel[0]  # We are only using the first element
    else:
        forces_motor = cmd_thrust

    # Creating force vector
    forces_motor_vec = cs.vertcat(0, 0, acc_coef + cmd_f_coef * forces_motor)

    # Linear equation of motion
    pos_dot = symbols.vel
    vel_dot = rot @ forces_motor_vec / mass + gravity_vec

    ddrpy = rpy_coef * symbols.rpy + rpy_rates_coef * symbols.drpy + cmd_rpy_coef * cmd_rpy

    if model_rotor_vel:
        X_dot = cs.vertcat(pos_dot, symbols.drpy, vel_dot, ddrpy, rotor_vel_dot, d_theta, cmd_dd_theta)
    else:
        X_dot = cs.vertcat(pos_dot, symbols.drpy, vel_dot, ddrpy, d_theta, cmd_dd_theta)
    Y = cs.vertcat(symbols.pos, symbols.rpy)

    return X_dot, X, U, Y
