"""This module implements the OCP formulation in ACADOS and creates the solver."""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from lsy_drone_racing.control.mpcc_model import symbolic_dynamics_euler_mpcc_so_rpy_rotor
import casadi as cs
from lsy_drone_racing.control.mpcc_solver_config import MPCCSolverConfig


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""

    mpcc_config = MPCCSolverConfig()
    X_dot, X, U = symbolic_dynamics_euler_mpcc_so_rpy_rotor(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        thrust_time_coef=parameters["thrust_time_coef"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    # MPCC Parameters

    M = mpcc_config.M
    theta_grid = mpcc_config.theta_grid

    p = cs.MX.sym(
        "p", 2 * 3 * M + 2 * 4 
    )  # für MPCC 9 + 8 (4 Hindernisse mit je 2D Position) + 16 (4 Gates mit je 4 Werten)
    model.p = p

    pd_list = p[: 3 * M]
    tp_list = p[3 * M : 2 * 3 * M]
    offset = 2 * M * 3
    # ---- Obstacle-Teil ----------------------------
    obs_1 = p[offset : offset + 2]
    obs_2 = p[offset + 2 : offset + 4]
    obs_3 = p[offset + 4 : offset + 6]
    obs_4 = p[offset + 6 : offset + 8]

    # # ---- Gate-Teil ----------------------------
    gates = p[offset + 8 :]

    # Extract variables from state / input
    position = X[0:3]
    attitude = X[6:9]
    control = U[:4]
    theta = X[-1]
    v_theta_cmd = U[-1]


    # Interpolate trajectory at current theta
    pd_theta = _piecewise_linear_interp(theta, theta_grid, pd_list)
    tp_theta = _piecewise_linear_interp(theta, theta_grid, tp_list)

    # Compute tracking errors
    tp_unit = tp_theta / (cs.norm_2(tp_theta) + 1e-6)
    e_theta = position - pd_theta
    e_lag = cs.dot(tp_unit, e_theta) * tp_unit  # Lag error (along path)
    e_contour = e_theta - e_lag  # Contour error (perpendicular)

    # MPCC Stage Cost

    Q_w = mpcc_config.q_attitude * cs.DM(np.eye(3))
    track_cost = (
        (mpcc_config.q_lag) * cs.dot(e_lag, e_lag)
        + (mpcc_config.q_contour) * cs.dot(e_contour, e_contour)
        + attitude.T @ Q_w @ attitude
    )

    # Control smoothness cost
    R_df = cs.DM(
        np.diag([mpcc_config.r_thrust, mpcc_config.r_roll, mpcc_config.r_pitch, mpcc_config.r_yaw])
    )
    smooth_cost = control.T @ R_df @ control

    # Speed incentive (maximize progress)
    speed_cost = -mpcc_config.mu_speed * v_theta_cmd

    stage_cost = track_cost + smooth_cost + speed_cost

    # Set cost expressions
    model.cost_expr_ext_cost = stage_cost

    # Obstacle-Constraint-Funktion
    r1 = 0.1**2 - ((position[0] - obs_1[0]) ** 2 + (position[1] - obs_1[1]) ** 2)
    r2 = 0.1**2 - ((position[0] - obs_2[0]) ** 2 + (position[1] - obs_2[1]) ** 2)
    r3 = 0.1**2 - ((position[0] - obs_3[0]) ** 2 + (position[1] - obs_3[1]) ** 2)
    r4 = 0.1**2 - ((position[0] - obs_4[0]) ** 2 + (position[1] - obs_4[1]) ** 2)

    # # Gate-Constraints (inside inner OR outside outer)
    # r_i = 0.10  # inner radius
    # r_o = 0.60  # outer radius
    # delta_gate = 0.30  # activation thickness along gate normal

    # gate_h_list = []
    # for i in range(4):
    #     base = 4 * i
    #     gx = gates[base + 0]
    #     gy = gates[base + 1]
    #     gz = gates[base + 2]
    #     psi = gates[base + 3]  # yaw

    #     dx_w = position[0] - gx
    #     dy_w = position[1] - gy
    #     dz_w = position[2] - gz

    #     c = cs.cos(psi)
    #     s = cs.sin(psi)

    #     # world -> gate frame (yaw-only wie im MPC)
    #     dx_g = c * dx_w + s * dy_w
    #     dy_g = -s * dx_w + c * dy_w
    #     dz_g = dz_w

    #     dist = dy_g * dy_g + dz_g * dz_g  # radius^2 in gate plane (y-z)

    #     # activate mainly near gate plane (dx_g ~ 0)
    #     w = (delta_gate**2) / (dx_g * dx_g + delta_gate**2)

    #     # ODER-Constraint: inside inner OR outside outer (never in between)
    #     h_gate = w * (dist - r_i**2) * (r_o**2 - dist)
    #     gate_h_list.append(h_gate)

    # gate_h = cs.vertcat(*gate_h_list)

    model.con_h_expr = cs.vertcat(r1, r2, r3, r4)

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()  # 14
    nu = ocp.model.u.rows()  # 5
    np_param = ocp.model.p.rows()  # 33

    ocp.dims.np = np_param
    ocp.parameter_values = np.zeros((np_param,))

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"
    # ocp.cost.cost_type_e = "EXTERNAL"

    # ----------- Constraint formulation ---------------

    # [f, f_cmd, r_cmd, p_cmd, y_cmd]
    thrust_min = parameters["thrust_min"] * 4
    thrust_max = parameters["thrust_max"] * 4
    ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Input constraints
    # [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd]
    ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
    ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 3.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # Obstacle constraints: BGH mit model.con_h_expr aus create_acados_model
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lh = np.array(4 * [-1e3])
    ocp.constraints.uh = np.zeros(4)
    ocp.constraints.idxsh = np.array([0, 1, 2, 3])
    nsbx = ocp.constraints.idxsh.shape[0]
    ocp.cost.Zl = 0 * np.ones((nsbx,))
    ocp.cost.Zu = np.array([1] * 4 )
    ocp.cost.zl = 0 * np.ones((nsbx,))
    ocp.cost.zu = np.array([1] * 4 )

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 100

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


def _piecewise_linear_interp(theta, theta_vec, flattened_points, dim: int = 3):
    """CasADi-friendly linear interpolation."""
    M = len(theta_vec)
    idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)

    idx_low = cs.floor(idx_float)
    idx_high = idx_low + 1
    alpha = idx_float - idx_low

    idx_low = cs.if_else(idx_low < 0, 0, idx_low)
    idx_high = cs.if_else(idx_high >= M, M - 1, idx_high)

    p_low = cs.vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
    p_high = cs.vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])

    return (1.0 - alpha) * p_low + alpha * p_high
