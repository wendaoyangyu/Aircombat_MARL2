import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import os
from envs.mae.mae_utils.base_env import BaseEnv


def traj_plot(env: BaseEnv, en: bool = True, 
              only_fig: bool = False, fig_path: str = None):
    if en:
        traj_plot_en(env)
    elif not en:
        traj_plot_zh(env, only_fig, fig_path)
    

def traj_plot_en(env: BaseEnv):
    # Maneuver Trajectory
    fig = plt.figure()
    # ax1 = fig.plot(projection='3d')
    ax1 = Axes3D(fig)  # 只会在matplotlib 3.5.1中生效
    # 这样只画出了scripts_to_agents中的entity
    for s in env.possible_scripts:
        a = env.scripts_to_agents[s]
        agent_idx = env._index_map[a]
        script_idx = env._scripts_map[s]

        red_array = env.agents_array[:, agent_idx, :]
        blue_array = env.scripts_array[:, script_idx, :]
        step_num = env.stop_steps[a]

        ax1.plot3D(red_array[0:step_num, 0], red_array[0:step_num, 1],
                    red_array[0:step_num, 2], color='r', label='trajectory of red UCAV')
        ax1.plot3D(blue_array[0:step_num, 0], blue_array[0:step_num, 1],
                    blue_array[0:step_num, 2], color='b', label='trajectory of blue UCAV')
        ax1.scatter3D(blue_array[0, 0], blue_array[0, 1], blue_array[0, 2], marker="*", label='o')
        ax1.scatter3D(red_array[0, 0], red_array[0, 1], red_array[0, 2], marker="o", label='p')
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('y/m')
    ax1.set_zlabel('z/m')
    ax1.legend()
    plt.title("Maneuver Trajectory")
    plt.show()

    # vertical view
    plt.figure()
    # 这样只画出了scripts_to_agents中的entity
    for s in env.possible_scripts:
        a = env.scripts_to_agents[s]
        agent_idx = env._index_map[a]
        script_idx = env._scripts_map[s]

        red_array = env.agents_array[:, agent_idx, :]
        blue_array = env.scripts_array[:, script_idx, :]
        step_num = env.stop_steps[a]

        plt.plot(red_array[0:step_num, 0], red_array[0:step_num, 1],
                    color='r', label='trajectory of red UCAV')
        plt.plot(blue_array[0:step_num, 0], blue_array[0:step_num, 1],
                    color='b', label='trajectory of blue UCAV')
        plt.scatter(blue_array[0, 0], blue_array[0, 1], marker="*", label='o')
        plt.scatter(red_array[0, 0], red_array[0, 1], marker="o", label='p')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.legend()
    plt.title("Maneuver Trajectory Vertical View")
    plt.show()


def traj_plot_zh(env: BaseEnv, only_fig: bool = False, fig_path: str = None):
    # Maneuver Trajectory
    fig = plt.figure(figsize=(8.5, 7))
    # ax1 = fig.plot(projection='3d')
    ax1 = Axes3D(fig)  # 只会在matplotlib 3.5.1中生效
    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # 指定红蓝的颜色
    r_color_arr = ['r', "DarkRed"]
    b_color_arr = ['b', "DarkBlue"]
    
    # 这样只画出了scripts_to_agents中的entity
    for s in env.possible_scripts:
        script_idx = env._scripts_map[s]
        agent_idx = script_idx

        red_array = env.agents_array[:, agent_idx, :]
        blue_array = env.scripts_array[:, script_idx, :]

        r_step_num = env.world.agents[agent_idx].fig_step + 1
        b_step_num = env.world.scripts[script_idx].fig_step + 1

        # ax1.plot3D(red_array[0:r_step_num, 0], red_array[0:r_step_num, 1],
        #            red_array[0:r_step_num, 2], color=r_color_arr[agent_idx], label=f'红方{agent_idx+1}号')
        ax1.plot3D(red_array[0:r_step_num, 0], red_array[0:r_step_num, 1],
                   red_array[0:r_step_num, 2], color=r_color_arr[agent_idx], label=f'Red {agent_idx+1}')
        # ax1.plot3D(blue_array[0:b_step_num, 0], blue_array[0:b_step_num, 1],
        #            blue_array[0:b_step_num, 2], color=b_color_arr[script_idx], label=f'蓝方{script_idx+1}号')
        ax1.plot3D(blue_array[0:b_step_num, 0], blue_array[0:b_step_num, 1],
                   blue_array[0:b_step_num, 2], color=b_color_arr[script_idx], label=f'Blue {script_idx+1}')
        if agent_idx == len(env.possible_agents) - 1:
            # ax1.scatter3D(red_array[0, 0], red_array[0, 1], red_array[0, 2], color='r', marker="o", label='红方起点', s=100)
            # ax1.scatter3D(blue_array[0, 0], blue_array[0, 1], blue_array[0, 2], color='b', marker="*", label='蓝方起点', s=100)
            ax1.scatter3D(red_array[0, 0], red_array[0, 1], red_array[0, 2], color='r', marker="o", label='Red Start', s=100)
            ax1.scatter3D(blue_array[0, 0], blue_array[0, 1], blue_array[0, 2], color='b', marker="*", label='Blue Start', s=100)
        else:
            ax1.scatter3D(red_array[0, 0], red_array[0, 1], red_array[0, 2], color='r', marker="o", s=100)
            ax1.scatter3D(blue_array[0, 0], blue_array[0, 1], blue_array[0, 2], color='b', marker="*", s=100)
    
    ticks_font = fm.FontProperties(family='Times New Roman')
    for tick in ax1.get_xticklabels():
        tick.set_fontproperties(ticks_font)
    for tick in ax1.get_yticklabels():
        tick.set_fontproperties(ticks_font)
    for tick in ax1.get_zticklabels():
        tick.set_fontproperties(ticks_font)
    ax1.tick_params(labelsize=20) # font size for all axis

    ax1.set_xlabel('X(m)', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=10)
    ax1.set_ylabel('Y(m)', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=15)
    ax1.set_zlabel('Z(m)', fontdict={'family': 'Times New Roman', 'size': 20}, labelpad=15)
    ax1.legend(prop={'family': 'Times New Roman', 'size': 14}, loc='upper left')
    plt.savefig(os.path.join(fig_path, '3d.png'))
    if only_fig is False:
        plt.show()

    # vertical view
    plt.figure(figsize=(8, 7))
    # 这样只画出了scripts_to_agents中的entity
    for s in env.possible_scripts:
        script_idx = env._scripts_map[s]
        agent_idx = script_idx

        red_array = env.agents_array[:, agent_idx, :]
        blue_array = env.scripts_array[:, script_idx, :]
        
        r_step_num = env.world.agents[agent_idx].fig_step + 1
        b_step_num = env.world.scripts[script_idx].fig_step + 1

        # plt.plot(red_array[0:r_step_num, 0], red_array[0:r_step_num, 1],
        #          color=r_color_arr[agent_idx], label=f'红方{agent_idx+1}号')
        plt.plot(red_array[0:r_step_num, 0], red_array[0:r_step_num, 1],
                 color=r_color_arr[agent_idx], label=f'Red {agent_idx+1}')
        # plt.plot(blue_array[0:b_step_num, 0], blue_array[0:b_step_num, 1],
        #          color=b_color_arr[script_idx], label=f'蓝方{script_idx+1}号')
        plt.plot(blue_array[0:b_step_num, 0], blue_array[0:b_step_num, 1],
                 color=b_color_arr[script_idx], label=f'Blue {script_idx+1}')
        if agent_idx == len(env.possible_agents) - 1:
            # plt.scatter(red_array[0, 0], red_array[0, 1], color='r', marker="o", label='红方起点', s=100)
            # plt.scatter(blue_array[0, 0], blue_array[0, 1], color='b', marker="*", label='蓝方起点', s=100)
            plt.scatter(red_array[0, 0], red_array[0, 1], color='r', marker="o", label='Red Start', s=100)
            plt.scatter(blue_array[0, 0], blue_array[0, 1], color='b', marker="*", label='Blue Start', s=100)
        else:
            plt.scatter(red_array[0, 0], red_array[0, 1], color='r', marker="o", s=100)
            plt.scatter(blue_array[0, 0], blue_array[0, 1], color='b', marker="*", s=100)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.xlabel('X(m)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.ylabel('Y(m)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.legend(prop={'family': 'Times New Roman', 'size': 14})
    plt.grid()
    plt.savefig(os.path.join(fig_path, '2d.png'))
    if only_fig is False:
        plt.show()


# def obs_plot(env: BaseEnv):
#     """
#     NOTE: 只针对cn和simple_2v2
#     TODO: 还未完成，暂时废弃
#     """
#     all_obs_array, true_steps_list = all_obs(env)
#     # 指定红蓝的颜色
#     r_color_arr = ['r', "DarkRed"]
#     b_color_arr = ['b', "DarkBlue"]
#     for i in range(len(env.possible_agents)):
#         plt.figure(figsize=(10, 7))
#         for j in range(len(env.possible_scripts)):
#             temp = i * len(env.possible_scripts) + j
#             step_num = true_steps_list[temp]
#             obs_array = all_obs_array[0:step_num, temp, :]
#             plt.plot(obs_array[0:step_num, 0], color='c', label='距离')  # , marker = '*', ms = 1.5
#         plt.ylabel('距离(m)', fontdict={'family': 'SimSun', 'size': 20})
#         plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 20})
#         plt.margins(x=0)
#         plt.tick_params(axis='both', labelsize=15)
#         plt.legend(prop={'family': 'SimSun', 'size': 18})
#         plt.show()
#         plt.figure(figsize=(10, 7))
#         for j in range(len(env.possible_scripts)):
#             temp = i * len(env.possible_scripts) + j
#             step_num = true_steps_list[temp]
#             obs_array = all_obs_array[0:step_num, temp, :]
#             plt.plot(obs_array[0:step_num, 3], color='r', label='天线偏转角')
#             plt.plot(obs_array[0:step_num, 4], color='b', label='天线方位角')
#         plt.ylabel('角度(°)', fontdict={'family': 'SimSun', 'size': 20})
#         plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 20})
#         plt.margins(x=0)
#         plt.tick_params(axis='both', labelsize=15)
#         plt.legend(loc='upper right', prop={'family': 'SimSun', 'size': 18})
#         plt.show()  


def obs_plot_0(env: BaseEnv, only_fig: bool = False, fig_path: str = None):
    all_obs_array, true_steps_list = all_obs(env)
    # distance
    plt.figure(figsize=(10, 7))
    for i in range(len(env.possible_agents)):
        for j in range(len(env.possible_scripts)):
            temp = i * len(env.possible_scripts) + j
            step_num = true_steps_list[temp]
            obs_array = all_obs_array[0:step_num, temp, :]
            # plt.plot(obs_array[0:step_num, 0], label=f'红{i+1}蓝{j+1}')  # 距离
            plt.plot(obs_array[0:step_num, 0], label=f'Red {i+1}-Blue {j+1}')  # 距离
    # plt.plot([1000]*max(true_steps_list), color='r', ls='--', label='胜利1')
    # plt.ylabel('距离(m)', fontdict={'family': 'SimSun', 'size': 25})
    # plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 25})
    plt.plot([1000]*max(true_steps_list), color='r', ls='--', label='Victory 1')
    plt.ylabel('Distance (m)', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.xlabel('Steps', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.margins(x=0)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.legend(prop={'family': 'Times New Roman', 'size': 14})
    plt.grid()
    plt.savefig(os.path.join(fig_path, 'dis.png'))
    if only_fig is False:
        plt.show()
    # ATA
    plt.figure(figsize=(10, 7))
    for i in range(len(env.possible_agents)):
        for j in range(len(env.possible_scripts)):
            temp = i * len(env.possible_scripts) + j
            step_num = true_steps_list[temp]
            obs_array = all_obs_array[0:step_num, temp, :]
            # plt.plot(obs_array[0:step_num, 3], label=f'红{i+1}蓝{j+1}')  # 天线偏转角
            plt.plot(obs_array[0:step_num, 3], label=f'Red {i+1}-Blue {j+1}')  # 天线偏转角
    # plt.plot([30]*max(true_steps_list), color='r', ls='--', label='胜利2')
    # plt.ylabel('角度(°)', fontdict={'family': 'SimSun', 'size': 25})
    # plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 25})
    plt.plot([30]*max(true_steps_list), color='r', ls='--', label='Victory 2')
    plt.ylabel('ATA-Angle (°)', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.xlabel('Steps', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.margins(x=0)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.legend(prop={'family': 'Times New Roman', 'size': 14})
    plt.grid()
    plt.savefig(os.path.join(fig_path, 'ata.png'))
    if only_fig is False:
        plt.show()
    # AA
    plt.figure(figsize=(10, 7))
    for i in range(len(env.possible_agents)):
        for j in range(len(env.possible_scripts)):
            temp = i * len(env.possible_scripts) + j
            step_num = true_steps_list[temp]
            obs_array = all_obs_array[0:step_num, temp, :]
            # plt.plot(obs_array[0:step_num, 4], label=f'红{i+1}蓝{j+1}')  # 天线方位角
            plt.plot(obs_array[0:step_num, 4], label=f'Red {i+1}-Blue {j+1}')  # 天线方位角
    # plt.plot([60]*max(true_steps_list), color='r', ls='--', label='胜利3')
    # plt.ylabel('角度(°)', fontdict={'family': 'SimSun', 'size': 25})
    # plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 25})
    plt.plot([60]*max(true_steps_list), color='r', ls='--', label='Victory 3')
    plt.ylabel('AA-Angle (°)', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.xlabel('Steps', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.margins(x=0)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.legend(prop={'family': 'Times New Roman', 'size': 14})
    plt.grid()
    plt.savefig(os.path.join(fig_path, 'aa.png'))
    if only_fig is False:
        plt.show()


def one_obs(red_array, blue_array, true_steps):
    """
    输出每个配对的obs_array
    """
    from ..mae_utils.scenario import BaseScenario
    sit_fun = BaseScenario().single_situation
    red_array = red_array[0:true_steps, :]
    blue_array = blue_array[0:true_steps, :]
    one_obs_array = np.zeros((true_steps, 6), dtype=float)
    for i in range(true_steps):
        obs = sit_fun(red_array[i], blue_array[i])
        one_obs_array[i] = obs
    return one_obs_array


def all_obs(env: BaseEnv):
    all_obs_len = len(env.possible_agents) * len(env.possible_scripts)
    all_obs_array = np.zeros((env.max_cycles+1, all_obs_len, 6), dtype=float)
    temp = 0
    true_steps_list = list()
    for a in env.possible_agents:
        for s in env.possible_scripts:
            agent_idx = env._index_map[a]
            script_idx = env._scripts_map[s]

            red_array = env.agents_array[:, agent_idx, :]
            blue_array = env.scripts_array[:, script_idx, :]
            
            r_step_num = env.world.agents[agent_idx].fig_step
            b_step_num = env.world.scripts[script_idx].fig_step
            true_steps = min(r_step_num, b_step_num) + 1
            true_steps_list.append(true_steps)

            one_obs_array = one_obs(red_array, blue_array, true_steps)
            all_obs_array[0:true_steps, temp, :] = one_obs_array
            temp += 1
    return all_obs_array, true_steps_list


def reward_plot(env: BaseEnv, only_fig: bool = False, fig_path: str = None):
    rewards_array = env.rewards_array
    rewards_array[rewards_array > 8] -= 10
    for a in env.possible_agents:
        agent_idx = env._index_map[a]
        step_num = env.world.agents[agent_idx].fig_step
        
        plot_ar = np.arange(1, step_num+1)
        x_ar = [1] + list(range(5, step_num+1, 5))

        plt.figure(figsize=(10, 7))
        plt.plot(plot_ar, rewards_array[0:step_num, agent_idx], color='r')  # 距离
        # plt.ylabel('奖励', fontdict={'family': 'SimSun', 'size': 25})
        # plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 25})
        plt.ylabel('Reward', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.xlabel('Steps', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.margins(x=0)
        plt.xticks(x_ar, fontproperties='Times New Roman', size=20)
        plt.yticks(fontproperties='Times New Roman', size=20)
        # plt.legend(prop={'family': 'SimSun', 'size': 22})
        plt.grid()
        plt.savefig(os.path.join(fig_path, f'traj_rew{agent_idx+1}.png'))
        if only_fig is False:
            plt.show()


def action_plot(env: BaseEnv, only_fig: bool = False, fig_path: str = None):
    # 避免交叉引用
    D2R = np.pi/180  # deg2rad
    R2D = 180/np.pi  # rad2deg
    NX_SCALE = 2
    NY_SCALE = 5
    GA_SCALE = np.pi  # math.pi/2.25

    ag_actions_array = env.ag_actions_array
    sc_actions_array = env.sc_actions_array
    # n_x
    for a in env.possible_agents:
        agent_idx = env._index_map[a]
        step_num = env.world.agents[agent_idx].fig_step
        
        plot_ar = np.arange(1, step_num+1)
        x_ar = [1] + list(range(5, step_num+1, 5))

        plt.figure(figsize=(10, 7))
        plt.plot(plot_ar, NX_SCALE * ag_actions_array[0:step_num, agent_idx, 0], color='r')  # 距离
        # plt.ylabel('切向过载(g)', fontdict={'family': 'SimSun', 'size': 25})
        # plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 25})
        plt.ylabel('Tangential Overload (g)', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.xlabel('Steps', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.margins(x=0)
        plt.xticks(x_ar, fontproperties='Times New Roman', size=20)
        plt.yticks(fontproperties='Times New Roman', size=20)
        # plt.legend(prop={'family': 'SimSun', 'size': 22})
        plt.grid()
        plt.savefig(os.path.join(fig_path, f'n_x{agent_idx+1}.png'))
        if only_fig is False:
            plt.show()
    # n_y
    for a in env.possible_agents:
        agent_idx = env._index_map[a]
        step_num = env.world.agents[agent_idx].fig_step
        
        plot_ar = np.arange(1, step_num+1)
        x_ar = [1] + list(range(5, step_num+1, 5))

        plt.figure(figsize=(10, 7))
        plt.plot(plot_ar, NY_SCALE * ag_actions_array[0:step_num, agent_idx, 1], color='r')  # 距离
        # plt.ylabel('法向过载(g)', fontdict={'family': 'SimSun', 'size': 25})
        # plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 25})
        plt.ylabel('Normal Overload (g)', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.xlabel('Steps', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.margins(x=0)
        plt.xticks(x_ar, fontproperties='Times New Roman', size=20)
        plt.yticks(fontproperties='Times New Roman', size=20)
        # plt.legend(prop={'family': 'SimSun', 'size': 22})
        plt.grid()
        plt.savefig(os.path.join(fig_path, f'n_y{agent_idx+1}.png'))
        if only_fig is False:
            plt.show()
    # gamma
    for a in env.possible_agents:
        agent_idx = env._index_map[a]
        step_num = env.world.agents[agent_idx].fig_step
        
        plot_ar = np.arange(1, step_num+1)
        x_ar = [1] + list(range(5, step_num+1, 5))

        plt.figure(figsize=(10, 7))
        plt.plot(plot_ar, R2D * GA_SCALE * ag_actions_array[0:step_num, agent_idx, 2], color='r')  # 距离
        # plt.ylabel('滚转角(°)', fontdict={'family': 'SimSun', 'size': 25})
        # plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 25})
        plt.ylabel('Roll angle(°)', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.xlabel('Steps', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.margins(x=0)
        plt.xticks(x_ar, fontproperties='Times New Roman', size=20)
        plt.yticks(fontproperties='Times New Roman', size=20)
        # plt.legend(prop={'family': 'SimSun', 'size': 22})
        plt.grid()
        plt.savefig(os.path.join(fig_path, f'gamma{agent_idx+1}.png'))
        if only_fig is False:
            plt.show()
    # b_action
    for b in env.possible_scripts:
        script_idx = env._scripts_map[b]
        step_num = env.world.scripts[script_idx].fig_step
        
        plot_ar = np.arange(1, step_num+1)
        x_ar = [1] + list(range(5, step_num+1, 5))
        
        plt.figure(figsize=(10, 7))
        plt.plot(plot_ar, sc_actions_array[0:step_num, script_idx], color='b')  # 距离
        # plt.ylabel('机动动作序号', fontdict={'family': 'SimSun', 'size': 25})
        # plt.xlabel('步数', fontdict={'family': 'SimSun', 'size': 25})
        plt.ylabel('Action Index', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.xlabel('Steps', fontdict={'family': 'Times New Roman', 'size': 25})
        plt.margins(x=0)
        plt.xticks(x_ar, fontproperties='Times New Roman', size=20)
        plt.yticks(fontproperties='Times New Roman', size=20)
        # plt.legend(prop={'family': 'SimSun', 'size': 22})
        plt.grid()
        plt.savefig(os.path.join(fig_path, f'action_b{script_idx+1}.png'))
        if only_fig is False:
            plt.show()
