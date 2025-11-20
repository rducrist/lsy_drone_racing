import casadi as ca
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosModel

# -------------------------------
# 1. Define the simple model
# -------------------------------
def export_simple_model():
    x = ca.SX.sym("x")
    u = ca.SX.sym("u")
    xdot = x + u  # explicit dynamics

    model = AcadosModel()
    model.x = ca.vertcat(x)
    model.u = ca.vertcat(u)
    model.f_expl_expr = ca.vertcat(xdot)   # use explicit dynamics
    # do NOT set f_impl_expr for explicit
    model.name = 'simple_system'
    return model

# -------------------------------
# 2. Set up the OCP
# -------------------------------
ocp = AcadosOcp()
ocp.model = export_simple_model()

# Linear least-squares cost
ocp.cost.cost_type = "LINEAR_LS"

nx = ocp.model.x.size()[0]  # 1
nu = ocp.model.u.size()[0]  # 1
ny = nx + nu                # 2

ocp.cost.yref = np.zeros(ny)      # stage cost reference
ocp.cost.yref_e = np.zeros(nx) 

# Weight matrices
Q_mat = 2 * np.diag([1.0])  # for state
R_mat = 2 * np.diag([2.0])  # for control

ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)  # stage cost
ocp.cost.W_e = Q_mat                                # terminal cost

# Mapping matrices: y = [x; u]
ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[0, 0] = 1.0

ocp.cost.Vu = np.zeros((ny, nu))
ocp.cost.Vu[1, 0] = 1.0

ocp.cost.Vx_e = np.eye(nx)  # terminal cost only on x

# -------------------------------
# 3. Constraints (optional)
# -------------------------------
# Here we keep it unconstrained
ocp.constraints.constr_type = 'BGH'
ocp.constraints.lbu = np.array([-1.0])
ocp.constraints.ubu = np.array([1.0])
ocp.constraints.idxbu = np.array([0])  # control bounds

# -------------------------------
# 4. Discretization
# -------------------------------
ocp.dims.N = 20        # prediction horizon
ocp.solver_options.tf = 2.0  # horizon length in seconds

# -------------------------------
# 5. Solver options
# -------------------------------
ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'

# -------------------------------
# 6. Create and solve
# -------------------------------
from acados_template import AcadosOcpSolver

ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

# initial condition
x0 = np.array([0.5])
ocp_solver.set(0, "lbx", x0)
ocp_solver.set(0, "ubx", x0)

# Solve OCP
status = ocp_solver.solve()
if status != 0:
    print("Acados solver returned status ", status)
else:
    x_opt = ocp_solver.get(0, "x")
    u_opt = ocp_solver.get(0, "u")
    print("Optimal x[0]:", x_opt)
    print("Optimal u[0]:", u_opt)
