"""Example MPC with attitude control for a quadrotor.

- Tracks an offline trajectory (sampled_trajectory.csv)
- Uses LINEAR_LS tracking cost
- Adds a soft inequality constraint h(x) >= 0:
    * h_gate(x) = r_gate^2 - ||pos - gate_pos||^2  >= 0
  => Drohne wird weich in eine Kugel um den Gate-Mittelpunkt gedrückt.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import os
import csv

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat  # <-- neu: für h(x) und Parameter

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Gewicht für Soft-h-Constraint (Gate-Kugel)
GATE_SOFT_WEIGHT = 1000.0  # hoch = "Gate-Zwang" stark


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model and adds h(x)-constraint."""
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

    # -------------------------------------------------------
    # Soft h(x)-Constraint: Kugel um aktuellen Gate-Mittelpunkt
    # Parametervektor p = [gx, gy, gz, r_gate]
    # h_gate = r_gate^2 - ((px-gx)^2 + (py-gy)^2 + (pz-gz)^2) >= 0
    # -------------------------------------------------------
    px = X[0]
    py = X[1]
    pz = X[2]

    p = MX.sym("p", 4)  # 0:gx,1:gy,2:gz,3:r_gate
    gx = p[0]
    gy = p[1]
    gz = p[2]
    r_gate = p[3]

    h_gate = r_gate**2 - ((px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2)

    model.con_h_expr = vertcat(h_gate)  # nh = 1
    model.p = p

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver with soft h(x) constraint."""
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

    # -------------------------------------------------------
    # Cost: Linear LS tracking
    # -------------------------------------------------------
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    Q = np.diag(
        [
            50.0,  # pos x
            50.0,  # pos y
            100.0,  # pos z
            1.0,  # roll
            1.0,  # pitch
            1.0,  # yaw
            10.0,  # vel x
            10.0,  # vel y
            10.0,  # vel z
            5.0,  # dr
            5.0,  # dp
            5.0,  # dy
        ]
    )
    R = np.diag(
        [
            1.0,  # roll cmd
            1.0,  # pitch cmd
            1.0,  # yaw cmd
            50.0,  # thrust cmd
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx:nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # -------------------------------------------------------
    # Hard box constraints (global)
    # -------------------------------------------------------
    ocp.constraints.idxbx = np.arange(nx, dtype=int)

    lbx = np.full(nx, -1e3)
    ubx = np.full(nx, 1e3)

    # Position noch grob begrenzen (Failsafe)
    lbx[0:3] = -2.0
    ubx[0:3] = 2.0

    # rpy-Bounds +/- 0.5 rad
    lbx[3:6] = -0.5
    ubx[3:6] = 0.5

    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx

    # Input constraints
    ocp.constraints.lbu = np.array(
        [
            -0.5,
            -0.5,
            -0.5,
            parameters["thrust_min"] * 4,
        ]
    )
    ocp.constraints.ubu = np.array(
        [
            0.5,
            0.5,
            0.5,
            parameters["thrust_max"] * 4,
        ]
    )
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # -------------------------------------------------------
    # (B) Soft h(x) constraint: Gate-Kugel
    # -------------------------------------------------------
    ocp.dims.nh = 1
    ocp.constraints.lh = np.array([0.0])       # h_gate >= 0
    ocp.constraints.uh = np.array([1e7])       # praktisch kein upper bound
    ocp.cost.W_soft = np.diag([GATE_SOFT_WEIGHT])  # 1x1-Matrix

    # Parameter: p = [gx, gy, gz, r_gate]
    ocp.dims.np = 4
    ocp.parameter_values = np.zeros(4)

    # -------------------------------------------------------
    # Initial state (wird zur Laufzeit überschrieben)
    # -------------------------------------------------------
    ocp.constraints.x0 = np.zeros((nx,))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # Prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file=None,  # build from Python OCP to avoid mismatches
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Trajectory-generating MPC using attitude control with soft gate h(x)-constraint."""
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._env_id = config.env.id

        self._N = 30
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Known gates
        self.gates = obs.get("gates_pos")  # shape: (num_gates, 3)
        self.gate_radius = 0.2             # geometrischer Gate-Radius
        # Weicher Kugelradius kann leicht größer sein:
        self.soft_gate_radius = 0.3

        self.current_gate_idx = 0

        # Obstacles (falls genutzt)
        self.obstacles = obs.get("obstacles_pos")

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )

        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._finished = False
        self._config = config

        # Optional: eigene State-Cost-Matrizen (nur Referenz, nicht direkt in OCP)
        self.Q_terminal = np.diag([500, 500, 500, 1, 1, 1, 10, 10, 10, 1, 1, 1])
        self.Q_state = np.diag([500, 500, 500, 1, 1, 1, 10, 10, 10, 1, 1, 1])

        # Offline trajectory
        self.traj_t = None
        self.traj_pos = None
        self.traj_vel = None
        self.traj_acc = None

        traj_path = "lsy_drone_racing/trajectories/sampled_trajectory.csv"
        self.load_offline_traj_from_csv(traj_path)
        self._time_since_traj_start = 0.0

        self._predicted_traj = None

    # ----------------- Trajectory loading -----------------

    def load_offline_traj_from_csv(self, csv_path: str) -> None:
        """Load a CSV with columns [t, px, py, pz, vx, vy, vz, ax, ay, az]."""
        t_list, p_list, v_list, a_list = [], [], [], []
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 10:
                    continue
                t = float(row[0])
                px, py, pz = float(row[1]), float(row[2]), float(row[3])
                vx, vy, vz = float(row[4]), float(row[5]), float(row[6])
                ax, ay, az = float(row[7]), float(row[8]), float(row[9])
                t_list.append(t)
                p_list.append([px, py, pz])
                v_list.append([vx, vy, vz])
                a_list.append([ax, ay, az])
        self.traj_t = np.array(t_list)
        self.traj_pos = np.array(p_list)
        self.traj_vel = np.array(v_list)
        self.traj_acc = np.array(a_list)
        if len(self.traj_t) >= 2:
            self._dt_offline = float(self.traj_t[1] - self.traj_t[0])
        else:
            self._dt_offline = None
        self.traj_loaded = True
        self.traj_viz = self.traj_pos[::100]

    def _traj_index_from_time(self, t_now: float) -> int:
        """Return index of trajectory time nearest and not earlier than t_now."""
        if not self.traj_loaded:
            return 0
        idx = np.searchsorted(self.traj_t, t_now, side="left")
        idx = max(0, min(idx, len(self.traj_t) - 1))
        return idx

    # ----------------- acc -> rpy/thrust -----------------

    def _acc_to_rpy_and_thrust(self, acc_des: np.ndarray, yaw_des: float = 0.0):
        """Convert desired acceleration (world frame) to desired roll, pitch, and thrust."""
        m = self.drone_params["mass"]
        g_vec = np.array(self.drone_params["gravity_vec"])  # e.g. [0,0,-9.81]
        F = m * (acc_des - g_vec)
        F_norm = np.linalg.norm(F)
        if F_norm < 1e-6:
            hover_thrust = m * -g_vec[-1]
            return 0.0, 0.0, yaw_des, hover_thrust

        b3 = F / F_norm
        phi = float(np.arctan2(b3[1], b3[2]))                 # roll
        theta = float(np.arctan2(-b3[0], np.sqrt(b3[1]**2 + b3[2]**2)))  # pitch
        thrust = float(F_norm)
        return phi, theta, yaw_des, thrust

    # ----------------- MPC step -----------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Computes the control command [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]."""

        # Current state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        # Fix state at stage 0
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Time & trajectory index
        t_now = self._time_since_traj_start
        self._time_since_traj_start += self._dt
        idx0 = self._traj_index_from_time(t_now)

        # Aktueller Gate-Mittelpunkt
        gate_pos = self.gates[self.current_gate_idx]  # shape (3,)
        gx, gy, gz = gate_pos
        r_gate = self.soft_gate_radius
        dist_ref_gate_end = np.linalg.norm(self.traj_pos[traj_idx_end] - gate_pos)
        if dist_ref_gate_end < 0.5:
            r_gate_T = self.soft_gate_radius
        else:
            r_gate_T = 5.0
        p_terminal = np.array([gx, gy, gz, r_gate_T])
        self._acados_ocp_solver.set(self._N, "p", p_terminal)

        # Build prediction horizon
        for k in range(self._N):
            traj_idx = min(idx0 + k, len(self.traj_t) - 1)
            pos_des = self.traj_pos[traj_idx]
            vel_des = self.traj_vel[traj_idx]
            acc_des = self.traj_acc[traj_idx]
            yaw_des = 0.0  # falls Trajektorie keinen Yaw enthält

            # ---- 2.1 Tracking-Referenz yref (states + inputs) ----
            yref = np.zeros(self._ny)

            # State-Anteil
            # Position
            yref[0:3] = pos_des
            # rpy aus gewünschter Beschleunigung
            r_des, p_des, y_des, thrust_ff = self._acc_to_rpy_and_thrust(acc_des, yaw_des)
            yref[3:6] = np.array([r_des, p_des, y_des])
            # Geschwindigkeit
            yref[6:9] = vel_des
            # drpy = 0 (oder später aus Trajektorie ableiten)

            # Input-Anteil (u-Referenz)
            yref[self._nx : self._nx + self._nu] = np.array([r_des, p_des, y_des, thrust_ff])

            self._acados_ocp_solver.set(k, "yref", yref)

            # ---- 2.2 Warm-Start für x und u ----
            x_guess = np.zeros(self._nx)
            x_guess[0:3] = pos_des
            x_guess[3:6] = np.array([r_des, p_des, y_des])
            x_guess[6:9] = vel_des
            self._acados_ocp_solver.set(k, "x", x_guess)
            self._acados_ocp_solver.set(k, "u", np.array([r_des, p_des, y_des, thrust_ff]))

            # ---- 2.3 Soft-h-Constraint Parameter p = [gx, gy, gz, r_gate_stage] ----
            # Distanz der Referenz zum Gate
            dist_ref_gate = np.linalg.norm(pos_des - gate_pos)

            # Wenn Referenz nah am Gate: "scharfe" Kugel, sonst riesige Kugel (quasi deaktiviert)
            if dist_ref_gate < 0.5:   # Schwelle z.B. 0.5 m
                r_gate_stage = self.soft_gate_radius   # z.B. 0.3–0.4 m
            else:
                r_gate_stage = 5.0    # sehr groß -> h_gate >= 0 fast immer erfüllt

            p_vec = np.array([gx, gy, gz, r_gate_stage], dtype=float)
            self._acados_ocp_solver.set(k, "p", p_vec)

        # -------------------------------------------------
        # 3) Terminal-Referenz + Terminal-Parameter p_N
        # -------------------------------------------------
        traj_idx_end = min(idx0 + self._N, len(self.traj_t) - 1)
        pos_des_end = self.traj_pos[traj_idx_end]

        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = pos_des_end
        yref_e[3:6] = np.array([0.0, 0.0, 0.0])
        yref_e[6:9] = np.zeros(3)
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Terminal-Soft-Constraint Parameter
        dist_ref_gate_end = np.linalg.norm(pos_des_end - gate_pos)
        if dist_ref_gate_end < 0.5:
            r_gate_T = self.soft_gate_radius
        else:
            r_gate_T = 5.0
        p_terminal = np.array([gx, gy, gz, r_gate_T], dtype=float)
        self._acados_ocp_solver.set(self._N, "p", p_terminal)

        # -------------------------------------------------
        # 4) MPC lösen + Fallback
        # -------------------------------------------------
        status = self._acados_ocp_solver.solve()
        if status != 0:
            print(f"[acados] solver returned status {status}, using hover fallback.")
            # Hover-Fallback statt u=0
            m = self.drone_params["mass"]
            g_vec = np.array(self.drone_params["gravity_vec"])
            hover_thrust = m * -g_vec[-1]
            return np.array([0.0, 0.0, 0.0, hover_thrust], dtype=float)

        u0 = self._acados_ocp_solver.get(0, "u")

        # Optional: predicted trajectory für Visualisierung
        predicted_positions = []
        for k in range(self._N + 1):
            x_pred = self._acados_ocp_solver.get(k, "x")
            predicted_positions.append(x_pred[:3])
        self._predicted_traj = np.array(predicted_positions)

        return u0

    # ----------------- Callbacks -----------------

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1

        # Gate als "erreicht" markieren, wenn nah genug
        if self.gates is not None:
            if np.linalg.norm(obs["pos"] - self.gates[self.current_gate_idx]) < self.gate_radius:
                self.current_gate_idx = min(self.current_gate_idx + 1, len(self.gates) - 1)

        return self._finished

    def episode_callback(self):
        self._tick = 0
        self.current_gate_idx = 0
        self._finished = False
        self._time_since_traj_start = 0.0
        self._predicted_traj = None
