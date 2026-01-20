"""This class generates plots from logged data."""


import matplotlib.pyplot as plt

from lsy_drone_racing.control.mpc_logger import MPCLogger


class MPCPlotter:
    """Plotter class for the controller. Might be used for MPC or MPCC."""
    def __init__(self, logger: MPCLogger):
        """Initialize the plotter with a logger."""
        self.log = logger

    def plot_solver_times(self):
        """Plot the OCP solver times."""
        plt.figure()
        plt.plot(self.log.solver_times)
        plt.xlabel("Iteration")
        plt.ylabel("Solver time [ms]")
        plt.title("Solver time per iteration")
        plt.grid(True)
        plt.show()

    def plot_costs(self):
        """Plot OCP costs."""
        plt.figure()
        plt.plot(self.log.costs)
        plt.xlabel("Iteration")
        plt.ylabel("MPC cost")
        plt.title("Cost vs iteration")
        plt.grid(True)
        plt.show()

    def plot_predictions(self, step: int = -1):
        """Plot the predicted trajectory at a given step."""
        if step < 0:
            step = len(self.log.predicted_trajs) + step

        traj = self.log.predicted_trajs[step]

        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1], marker="o")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Predicted trajectory at step {step}")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
