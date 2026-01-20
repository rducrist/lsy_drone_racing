"""This module implements the OCP formulation in ACADOS and creates the solver."""

from __future__ import annotations  # Python 3.10 type hints

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from lsy_drone_racing.control.mpcc_model import so_rpy_rotor_drag_first_order
from lsy_drone_racing.control.mpcc_solver_config import MPCCSolverConfig


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    mpcc_config = MPCCSolverConfig()
    X_dot, X, U = so_rpy_rotor_drag_first_order(
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
        drag_linear_coef = parameters["drag_linear_coef"], 
        drag_square_coef = parameters["drag_square_coef"]
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
        "p", 2 * 3 * M + M + 2 * 4 # tp_list, pd_list, q_dyn, obstacles 
    )  
    model.p = p

    pd_list = p[0:3*M]
    tp_list = p[3*M:6*M]
    qc_dyn = p[6*M:7*M]
    offset = 2 * M * 3 + M
   
   # Parse obstacles
    obs_1 = p[offset : offset + 2]
    obs_2 = p[offset + 2 : offset + 4]
    obs_3 = p[offset + 4 : offset + 6]
    obs_4 = p[offset + 6 : offset + 8]


    # Extract variables from state / input
    position = X[0:3]
    attitude = X[6:9]
    control = U[:4]
    theta = X[-1]
    v_theta_cmd = U[-1]

    # Interpolate trajectory at current theta
    pd_theta = _piecewise_linear_interp(theta, theta_grid, pd_list)
    tp_theta = _piecewise_linear_interp(theta, theta_grid, tp_list)
    qc_dyn_theta = _piecewise_linear_interp(theta, theta_grid, qc_dyn, dim=1)

    # Compute tracking errors
    tp_unit = tp_theta / (cs.norm_2(tp_theta) + 1e-6)
    e_theta = position - pd_theta
    e_lag = cs.dot(tp_unit, e_theta) * tp_unit  # Lag error (along path)
    e_contour = e_theta - e_lag  # Contour error (perpendicular)

    # MPCC Cost Formulation

    Q_w = mpcc_config.q_attitude * cs.DM(np.eye(3))
    
    # Lag
    C_l = mpcc_config.q_lag + mpcc_config.q_lag_peak * qc_dyn_theta
    lag_cost = C_l * cs.dot(e_lag, e_lag)

    #Contour
    C_c = mpcc_config.q_contour + mpcc_config.q_contour_peak * qc_dyn_theta
    contour_cost = C_c * cs.dot(e_contour, e_contour)

    #Attitude
    attitude_cost = attitude.T @ Q_w @ attitude

    # Control smoothness cost
    R_df = cs.DM(
        np.diag([mpcc_config.r_roll, mpcc_config.r_pitch, mpcc_config.r_yaw, mpcc_config.r_thrust])
    )
    smooth_cost = control.T @ R_df @ control

    # Speed incentive (maximize progress)
    speed_cost = -mpcc_config.mu_speed * v_theta_cmd

    stage_cost = lag_cost + contour_cost + attitude_cost + smooth_cost + speed_cost

    # Set cost expressions
    model.cost_expr_ext_cost = stage_cost

    # Obstacle Constraints
    r1 = 0.2**2 - ((position[0] - obs_1[0]) ** 2 + (position[1] - obs_1[1]) ** 2)
    r2 = 0.2**2 - ((position[0] - obs_2[0]) ** 2 + (position[1] - obs_2[1]) ** 2)
    r3 = 0.2**2 - ((position[0] - obs_3[0]) ** 2 + (position[1] - obs_3[1]) ** 2)
    r4 = 0.2**2 - ((position[0] - obs_4[0]) ** 2 + (position[1] - obs_4[1]) ** 2)

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
    nx = ocp.model.x.rows()  
    np_param = ocp.model.p.rows() 

    ocp.dims.np = np_param
    ocp.parameter_values = np.zeros((np_param,))

    # Set dimensions
    ocp.solver_options.N_horizon = N

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"

    # ----------- Constraint formulation ---------------

    # [f, f_cmd, r_cmd, p_cmd, y_cmd]
    thrust_min = parameters["thrust_min"] * 4
    thrust_max = parameters["thrust_max"] * 4
    ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 13, 10, 11, 12])

    # Input constraints
    # [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd]
    ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
    ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0,1.3])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # Soft Obstacle Constraints
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lh = np.array(4 * [-1e3])
    ocp.constraints.uh = np.zeros(4)
    ocp.constraints.idxsh = np.array([0, 1, 2, 3])
    nsbx = ocp.constraints.idxsh.shape[0]
    ocp.cost.Zl = 0 * np.ones((nsbx,))
    ocp.cost.Zu = np.array([200] * 4 )
    ocp.cost.zl = 0 * np.ones((nsbx,))
    ocp.cost.zu = np.array([200] * 4 )

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
    ocp.solver_options.nlp_solver_max_iter = 10
    # ocp.solver_options.regularize_method = "PROJECT"

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


def _piecewise_linear_interp(theta, theta_vec, flattened_points, dim: int = 3):  # noqa: ANN001, ANN202 # No types because casadi types are hard to handle
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
