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
        px, py, pz, vx, vy, vz, roll, pitch, yaw, f_collective, f_cmd, r_cmd, p_cmd, y_cmd, theta
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd)

    # Dynamics equations
    thrust = f_collective
    inv_mass = 1.0 / mass

    # Acceleration from thrust
    ax = inv_mass * thrust * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw))
    ay = inv_mass * thrust * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw))
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
        v_theta_cmd,
    )

    return f_dyn, states, inputs


def symbolic_dynamics_euler(
    model_rotor_vel: bool = True,
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
    drag_linear_coef: Array,
    drag_square_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """The fitted linear, second order rpy dynamics with thrust dynamics and drag.

    For info on the args, see above.

    This function returns the actual model, as defined in the paper, for direct use.
    """
    # States and Inputs
    f_collective = symbols.rotor_vel[0]
    cmd_dthrust = cs.MX.sym("cmd_dthrust")
    cmd_drpy = cs.MX.sym("cmd_drpy", 3)
    theta = cs.MX.sym("theta")
    cmd_v_theta = cs.MX.sym("cmd_v_theta")
    X = cs.vertcat(
        symbols.pos, symbols.vel,  symbols.rpy, symbols.drpy, f_collective, symbols.cmd_rpyt, theta
    )
    U = cs.vertcat(cmd_drpy, cmd_dthrust, cmd_v_theta)
    cmd_thrust = X[16]
    cmd_rpy = X[13:16]
    cmd_drpy = U[:3]
    cmd_dthrust = U[3]
    rot = rotation.cs_rpy2matrix(symbols.rpy)

    # Defining the dynamics function
    # Note that we are abusing the rotor_vel state as the thrust
    rotor_vel_dot = 1 / thrust_time_coef * (cmd_thrust - symbols.rotor_vel[0])
    forces_motor = f_collective  # We are only using the first element

    # Creating force vector
    forces_motor_vec = cs.vertcat(0, 0, acc_coef + cmd_f_coef * forces_motor)

    # Linear equation of motion
    pos_dot = symbols.vel
    vel_dot = (
        rot @ forces_motor_vec / mass
        + gravity_vec
        + 1 / mass * drag_linear_coef * symbols.vel
        + 1 / mass * drag_square_coef * symbols.vel * cs.fabs(symbols.vel)
    )

    ddrpy = rpy_coef * symbols.rpy + rpy_rates_coef * symbols.drpy + cmd_rpy_coef * cmd_rpy

    X_dot = cs.vertcat(
        pos_dot, vel_dot, symbols.drpy, ddrpy, rotor_vel_dot[0], cmd_drpy, cmd_dthrust, cmd_v_theta
    )
    Y = cs.vertcat(symbols.pos, symbols.rpy)

    return X_dot, X, U

def so_rpy_rotor_drag_first_order(
    model_rotor_vel: bool = True,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,   # unused but kept for signature compatibility
    cmd_rpy_coef: Array,
    drag_linear_coef: Array,
    drag_square_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:

    # --- States ---
    f_collective = symbols.rotor_vel[0]
    theta = cs.MX.sym("theta")

    X = cs.vertcat(
        symbols.pos,        # 0:3
        symbols.vel,        # 3:6
        symbols.rpy,        # 6:9
        f_collective,       # 9
        symbols.cmd_rpyt,   # 10:14 (r_cmd, p_cmd, y_cmd, f_cmd)
        theta               # 14
    )

    # --- Inputs ---
    cmd_drpy = cs.MX.sym("cmd_drpy", 3)
    cmd_dthrust = cs.MX.sym("cmd_dthrust")
    cmd_v_theta = cs.MX.sym("cmd_v_theta")

    U = cs.vertcat(cmd_drpy, cmd_dthrust, cmd_v_theta)

    # --- Command extraction ---
    cmd_rpy = X[10:13]
    f_cmd   = X[13]

    rot = rotation.cs_rpy2matrix(symbols.rpy)

    # --- Thrust dynamics ---
    df_collective = (f_cmd - f_collective) / thrust_time_coef

    # --- Force model ---
    forces_motor_vec = cs.vertcat(
        0,
        0,
        acc_coef + cmd_f_coef * f_collective
    )

    # --- Translational dynamics ---
    pos_dot = symbols.vel

    vel_dot = (
        rot @ forces_motor_vec / mass
        + gravity_vec
        + drag_linear_coef / mass * symbols.vel
        + drag_square_coef / mass * symbols.vel * cs.fabs(symbols.vel)
    )

    # --- FIRST-ORDER RPY DYNAMICS ---
    # rpy_dot = A * rpy + B * rpy_cmd

    rpy_coef_fo     = -rpy_coef / rpy_rates_coef
    cmd_rpy_coef_fo = -cmd_rpy_coef / rpy_rates_coef

    rpy_dot = rpy_coef_fo * symbols.rpy + cmd_rpy_coef_fo * cmd_rpy

    # acc_coef = 0.0
    # cmd_f_coef = 0.98023254
    # thrust_time_coef = 0.07993871
    # drag_linear_coef = -0.02149163
    # drag_square_coef = -0.02359736
    # rpy_coef = [-188.9910, -188.9910, -138.3109]
    # rpy_rates_coef = [-12.7803, -12.7803, -16.8485]
    # cmd_rpy_coef = [138.0834, 138.0834, 198.5161]

    # --- Command integrators ---
    cmd_rpy_dot = cmd_drpy
    f_cmd_dot   = cmd_dthrust

    # --- State derivative ---
    X_dot = cs.vertcat(
        pos_dot,
        vel_dot,
        rpy_dot,
        df_collective,
        cmd_rpy_dot,
        f_cmd_dot,
        cmd_v_theta
    )

    Y = cs.vertcat(symbols.pos, symbols.rpy)

    return X_dot, X, U