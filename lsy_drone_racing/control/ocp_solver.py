"""This module implements the OCP formulation in ACADOS and creates the solver."""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.so_rpy import symbolic_dynamics_euler
import casadi as cs


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    X_dot, X, U, _ = symbolic_dynamics_euler(
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
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            30.0,  # pos
            30.0,  # pos
            100.0,  # pos
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            50.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # thrust
        ]
    )

    # -------- Cost formulation --------------------
    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # ----------- Constraint formulation ---------------

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-1e3, -1e3, -1e3, -0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([1e3, 1e3, 1e3, 0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Set obstacle constraints
    n_gates = 4
    n_obs = 4

    gate_c = cs.MX.sym("gate_c", 4, n_gates)   # 4 (pos + yaw) × n_gates
    obs_c  = cs.MX.sym("obs_c", 2, n_obs)     # 2 × n_obs

    ocp.model.p = cs.vertcat(
    cs.reshape(gate_c, -1, 1),
    cs.reshape(obs_c, -1, 1),)

    ocp.parameter_values = np.zeros((4*n_gates + 2*n_obs,))

    # obstacles 
    r_obs = 0.4
    obs_h_list = []

    for i in range(n_obs):
        dx = ocp.model.x[0] - obs_c[0, i]
        dy = ocp.model.x[1] - obs_c[1, i]
        dist = cs.sqrt(dx**2 + dy**2 + 1e-6)
        h_obs = (r_obs - dist) / r_obs
        obs_h_list.append(h_obs)

    obs_h = cs.vertcat(*obs_h_list)
    
    # gates 
    r_i = 0.1
    r_o = 0.5
    gate_scale = r_o - r_i
    delta = 1.0
    # Activation parameters
    activation_x = 0.2  # Activate ±1.5m from gate plane
    activation_r = 0.5  # Activate within 1m of gate center axis

    gate_h_list = []

    for i in range(n_gates):

        # gate parameters
        gx = gate_c[0, i]
        gy = gate_c[1, i]
        gz = gate_c[2, i]
        psi = gate_c[3, i]   # yaw angle

        # relative position in world frame
        dx_w = ocp.model.x[0] - gx
        dy_w = ocp.model.x[1] - gy
        dz_w = ocp.model.x[2] - gz

        # rotation world -> gate frame
        cos_psi = cs.cos(psi)
        sin_psi = cs.sin(psi)

        dx_g =  cos_psi * dx_w + sin_psi * dy_w
        dy_g = -sin_psi * dx_w + cos_psi * dy_w
        dz_g = dz_w

         # Distance from gate center axis
        r = cs.sqrt(dy_g**2 + dz_g**2 + 1e-6)

        # Activation: near gate plane AND near gate axis
        weight_x = activation_x**2 / (dx_g**2 + activation_x**2)
        weight_r = activation_r**2 / (r**2 + activation_r**2)
        weight = weight_x * weight_r

        # Normalized complementarity form
        h_gate = weight * (r - r_i) * (r_o - r) / gate_scale**2
        
        gate_h_list.append(h_gate)

    gate_h = cs.vertcat(*gate_h_list)

    ocp.model.con_h_expr = cs.vertcat(
    gate_h,
    obs_h)

    ocp.constraints.constr_type = "BGH"

    nh = n_gates + n_obs

    ocp.constraints.lh = -1e3 * np.ones(nh)
    ocp.constraints.uh = 0 * np.ones(nh)

    ocp.constraints.idxsh = np.arange(nh)

    ocp.cost.Zl = 0 * np.ones(nh)
    ocp.cost.Zu = np.array([2500]*n_gates + [500]*n_obs)
    ocp.cost.zl = 0 * np.ones(nh)
    ocp.cost.zu = np.array([1]*n_gates + [1]*n_obs)

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

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
