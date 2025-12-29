"""This module implements the OCP formulation in ACADOS and creates the solver."""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from lsy_drone_racing.control.mpcc_model import symbolic_dynamics_euler_mpcc
import casadi as cs


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    X_dot, X, U, _ = symbolic_dynamics_euler_mpcc(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
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

    #MPCC Parameters
    p = cs.MX.sym("p", 17) #für MPCC 9 + 8 (4 Hindernisse mit je 2D Position)
    model.p = p
    
    p_ref = p[0:3]
    t_ref = p[3:6]
    qc    = p[6]
    ql    = p[7]
    mu    = p[8]

    # ---- Obstacle-Teil ----------------------------
    obs_1 = p[9:11]
    obs_2 = p[11:13]
    obs_3 = p[13:15]
    obs_4 = p[15:17]

    #Extract variables from state / input 
    pos    = X[0:3]
    vtheta = X[-1]       # last state
    dvtheta_cmd = U[4]
    u_rpy  = U[0:3]
    u_T    = U[3]

    #MPCC error terms el, ec
    delta = pos - p_ref
    lag_err = cs.dot(delta, t_ref)
    cont_err = delta - lag_err * t_ref

    ec_sq = cs.dot(cont_err, cont_err)
    el_sq = lag_err**2

    #Regularitaion weights 
    R_vth = parameters.get("R_vtheta", 3.2) # weight for smooth progress accel
    R_u   = parameters.get("R_inputs", 1.0) # weight for control inputs rpy
    R_T   = parameters.get("R_thrust", 11.0) # weight for thrust 

    #MPCC Stage Cost

    stage_cost = (
        qc * ec_sq +            # contouring
        ql * el_sq +            # lag
        R_vth * dvtheta_cmd**2 +# smooth progress accel
        R_u   * cs.dot(u_rpy, u_rpy) +
        R_T   * u_T**2 -
        mu * vtheta             # progress reward
    )
    
    # Terminal cost (no inputs)
    terminal_cost = qc * ec_sq + ql * el_sq

    # Set cost expressions
    model.cost_expr_ext_cost   = stage_cost
    model.cost_expr_ext_cost_e = terminal_cost

    # Obstacle-Constraint-Funktion (wenn du BGH behalten willst)
    r1 = 0.15**2 - ( (pos[0]-obs_1[0])**2 + (pos[1]-obs_1[1])**2 )
    r2 = 0.15**2 - ( (pos[0]-obs_2[0])**2 + (pos[1]-obs_2[1])**2 )
    r3 = 0.15**2 - ( (pos[0]-obs_3[0])**2 + (pos[1]-obs_3[1])**2 )
    r4 = 0.15**2 - ( (pos[0]-obs_4[0])**2 + (pos[1]-obs_4[1])**2 )

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
    nx = ocp.model.x.rows() # 14
    nu = ocp.model.u.rows() # 5
    np_param = ocp.model.p.rows() # 17
    
    ocp.dims.np= np_param
    ocp.parameter_values = np.zeros((np_param,))

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"


    # ----------- Constraint formulation ---------------

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-1e3, -1e3, -1e3, -0.9, -0.9, -0.9])
    ocp.constraints.ubx = np.array([1e3, 1e3, 1e3, 0.9, 0.9, 0.9])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5])

    # Set Input Constraints 
    # (rpy < 30°) and (thrust within physical limits) and (dvtheta_cmd limits)
    dvtheta_min = -10.0
    dvtheta_max =  10.0
    ocp.constraints.lbu = np.array([-0.9, -0.9, -0.9, parameters["thrust_min"] * 4, dvtheta_min])
    ocp.constraints.ubu = np.array([0.9, 0.9, 0.9, parameters["thrust_max"] * 4, dvtheta_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])


    # Obstacle constraints: BGH mit model.con_h_expr aus create_acados_model
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lh = np.array([-1e3, -1e3, -1e3, -1e3])
    ocp.constraints.uh = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.constraints.idxsh = np.array([0, 1, 2, 3])
    nsbx = ocp.constraints.idxsh.shape[0]
    ocp.cost.Zl = 5 * np.ones((nsbx,))
    ocp.cost.Zu = 5 * np.ones((nsbx,))
    ocp.cost.zl = 1 * np.ones((nsbx,))
    ocp.cost.zu = 350 * np.ones((nsbx,))

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-4

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

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
