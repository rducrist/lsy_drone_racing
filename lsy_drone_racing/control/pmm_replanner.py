from pmm_planner.utils import plan_pmm_trajectory
import numpy as np
import matplotlib.pyplot as plt

def debug_plot(p_s):
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(p_s[:, 0], p_s[:, 1], p_s[:, 2])
    plt.show()


waypoints_config = {
    "start_velocity": [0, 0, 0],
    "end_velocity": [0, 0, 0],
    "waypoints":  np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        ),
}
# with waypoints dict
planner_config_file = "./pmm_uav_planner/config/planner/crazyflie.yaml"
traj = plan_pmm_trajectory(waypoints_config, planner_config_file)

sampling_period = 0.05
t_s, p_s, v_s, a_s = traj.get_sampled_trajectory(sampling_period)
t_s, p_s, v_s, a_s = np.array(t_s), np.array(p_s), np.array(v_s), np.array(a_s)
debug_plot(p_s)