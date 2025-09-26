import numpy as np
from gym.utils import EzPickle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO

from envs.mae_parallel.mae_utils.core import Agent, Script, World
from envs.mae_parallel.mae_utils.scenario import BaseScenario
from envs.mae_parallel.mae_utils.base_env import BaseEnv

"""
    simple_1v1:
    1. red: agent, steady
    2. blue: script, mini_max

    del:
    1. "max_step" -> max_cycles √
    2. "switch" -> X √
    3. "tau" -> 放入world() √
    4. "action_b_mode" -> scenario
    5. "regular_b_name" -> scenario
    6. "simu_r" -> scenario
    7. "simu_b" -> scenario
    8. "nx_s" -> 固定 √
    9. "ny_s" -> 固定 √
    10. "gamma_s" -> 固定 √
    11. "z_limit" -> X √

    NOTE:
    1. scenario: 
    - 设置了环境的具体agents个数、类型等
    - 设置了环境相关的reward、observation(因为这和任务相关)
    - 必须首先设置close
    2. core: 
    - 设置了agents的具体属性和动力学模型
    3. simple_env: 
    - 设置了agents的具体动作(由外部输入)
    - TODO: scripts的具体动作也应该由这里设置

TODO: 
    1. 添加total_observation
    2. 采用scripts完成脚本型aircraft运动，  
        scripts只支持离散动作
    3. 应该覆写simple_env.step()的done条件
"""

class raw_env(BaseEnv, EzPickle):
    """
    该类是simple_adversary的指定class

    simple_adversary的具体实现步骤:
    1. scenario -> 
    2. 设置world参数 —> 
    3. BaseEnv(包含mpe系列环境的具体attr)

    DO:
    1. 将max_cycles传入scenario
    """
    def __init__(self, N=1, max_cycles=200, continuous_actions=False, render_mode=None, scr_mode='mini_max'):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )  # 初始化一些参数
        scenario = Scenario(N, max_cycles)  # 实例化脚本
        world = scenario.make_world()  # 通过脚本设置core.World中的一些参数
        BaseEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )  # 将设置好的world等传入BaseEnv基类中进行初始化
        self.metadata["name"] = "simple_1v1"

    def monitor(self):
        step_num = self.stop_steps.get('red_0')
        count = self.max_cycles + 1
        # red_array = self.agents_array[:, 0, :].reshape(count, 6)
        red_array = self.agents_array[:, 0, :]  # 二维数组
        blue_array = self.scripts_array[:, 0, :]

        actions_r_array = self.ag_actions_array[:, 0, :]
        actions_b_array = self.sc_actions_array[:, 0, :]
        
        obs_array = np.zeros((count, 6), dtype=float)
        for i in range(step_num):
            red_sim = red_array[i]
            blue_sim = blue_array[i]
            obs_array[i] = self.scenario.single_situation(red_sim, blue_sim)
        
        # Maneuver Trajectory
        fig = plt.figure()
        # ax1 = fig.plot(projection='3d')
        ax1 = Axes3D(fig)
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
        # obs = [distance, delta_h, delta_v, q_r, q_b, beta_rb]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)  # distance
        # plt.scatter(obs_array[0:step_num, 0], color='c', label='distance')
        plt.tick_params(axis="both", which="minor")
        plt.plot(obs_array[0:step_num, 0], color='c', label='distance')  # , marker = '*', ms = 1.5
        plt.plot([1000] * step_num, color='y', ls='--', label='termination1')
        plt.plot([100] * step_num, color='y', ls='--', label='termination2')
        plt.ylabel('distance/m')
        plt.xlabel('step')
        plt.margins(x=0)
        plt.legend()
        plt.subplot(1, 2, 2)  # q_r
        plt.tick_params(axis="both", which="minor")
        plt.plot(obs_array[0:step_num, 3], color='r', label='ATA')
        plt.plot(obs_array[0:step_num, 4], color='b', label='AA')
        plt.plot([30]*step_num, color='r', ls=':', label='ATA(red_win)')
        plt.plot([60]*step_num, color='r', ls='--', label='AA(red_win)')
        plt.plot([120]*step_num, color='b', ls=':', label='ATA(blue_win)')
        plt.plot([150]*step_num, color='b', ls='--', label='AA(blue_win)')
        plt.ylabel('angle/deg')
        plt.xlabel('step')
        plt.margins(x=0)
        plt.legend()
        plt.suptitle("red vs blue")
        plt.show()
        if self.continuous_actions is False:
            # Maneuver Selection
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(actions_r_array[0:step_num])
            plt.ylim(1, 7)
            plt.grid()
            plt.ylabel('Red Action Order')
            plt.margins(x=0)
            plt.subplot(2, 1, 2)
            plt.plot(actions_b_array[0:step_num])
            plt.ylim(1, 7)
            plt.grid()
            plt.ylabel('Blue Action Order')
            plt.xlabel('step')
            plt.margins(x=0)
            plt.suptitle("Maneuver Selection")
            plt.show()
        else:
            assert self.continuous_actions is True
            R2D = 180 / np.pi  # rad2deg
            NX_SCALE = self.world.constant.NX_SCALE
            NY_SCALE = self.world.constant.NY_SCALE
            GA_SCALE = self.world.constant.GA_SCALE
            # red Maneuver Selection
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(NX_SCALE * actions_r_array[0:step_num, 0])
            plt.ylim(-2, 2)
            plt.ylabel('n_x/g')
            plt.margins(x=0)
            plt.grid()
            plt.subplot(3, 1, 2)
            plt.plot(NY_SCALE * actions_r_array[0:step_num, 1])
            plt.ylim(-5, 5)
            plt.ylabel('n_y/g')
            plt.margins(x=0)
            plt.grid()
            plt.subplot(3, 1, 3)
            plt.plot(R2D * GA_SCALE * actions_r_array[0:step_num, 2])
            plt.ylim(-R2D * GA_SCALE, R2D * GA_SCALE)
            plt.ylabel('gamma/deg')
            plt.xlabel('step')
            plt.margins(x=0)
            plt.grid()
            plt.suptitle("Red Overload Changes")
            plt.show()
            # blue Maneuver Selection
            plt.figure(figsize=(8, 4))
            plt.plot(actions_b_array[0:step_num])
            plt.ylim(1, 7)
            plt.grid()
            plt.ylabel('Blue Action Order')
            plt.xlabel('step')
            plt.margins(x=0)
            plt.suptitle("Blue Maneuver Selection")
            plt.show()


class Scenario(BaseScenario):
    """
    设置world的具体脚本:
    1. 通过make_world初始化一个world
    2. 通过reset_world进行位置和速度矢量初始化
    """

    def __init__(self, N=1, max_cycles=400):
        """
        将N、max_cycles作为实例化时传入
        """
        super().__init__()
        self.N = N
        self.max_cycles = max_cycles

    def make_world(self):
        """
        make_world完成了以下:
        1. 设置agents、scripts的个数
        2. 为agent添加了name
        3. 为script添加了name
        """
        world = World()  # 实例化world
        # set any world properties first
        num_agents = 1
        world.num_agents = num_agents
        num_script = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]  # 根据num放置core.Agent()
        for i, agent in enumerate(world.agents):
            # * core.Agent()中并没有adversary这个属性
            base_name = "red"
            base_index = i if i < num_agents else i - num_agents
            agent.name = f"{base_name}_{base_index}"  # 前两行均是为其做准备
        # add scripts
        world.scripts = [Script() for i in range(num_script)]
        for i, script in enumerate(world.scripts):
            script.name = "blue_%d" % i
        return world

    def reset_world(self, world: World, np_random):
        """
        reset_world完成了以下:
        1. 初始化agents、scripts的位置和速度
            ！这里不随机化
        2. TODO: 这里的np_random for seed是否生效未知
        """
        # set is_alive
        for agent in world.agents:
            agent.is_alive = True
        for script in world.scripts:
            script.is_alive = True
        # set initial states
        # red_0
        r0 = world.agents[0]
        r0.initial_state(np.array([0, 0, 3000, 45, 0, 200]))
        # blue_0
        b0 = world.scripts[0]
        b0.initial_state(np.array([3000, 3000, 3000, 90, 0, 220]))

    def reward(self, agent: Agent, env: BaseEnv):
        """
        Reward for close 1v1.

        NOTE:
        1. 将输入参数中的agent改为entity
        2. agent, script必须分开获得reward
        3. 输入参数的形式不好改变
        4. entity, world中都不含步数信息，所有还需要一个步数作为输入参数
        """
        script = env.world.scripts[0]
        exp = np.exp
        last_win_num = env.win_num
        last_lose_num = env.lose_num
        if agent.is_alive:
            # red
            x_r, y_r, z_r, phi_r, mu_r, v_r = agent.unpack
            x_b, y_b, z_b, phi_b, mu_b, v_b = script.unpack 
            # 获取态势
            obs = self.single_situation(agent.unpack, script.unpack)  # distance, delta_h, delta_v, q_r, q_b
            distance = obs[0]
            delta_h = obs[1]
            delta_v = obs[2]
            q_r = obs[3]
            q_b = obs[4]
            beta_rb = obs[5]
            # 负面奖励
            r_p = 0
            if z_r <= 2000 or z_r >= 20000:
                r_p = -10
            elif v_r <= 100 or v_r >= 500:
                r_p = -10
            else:
                r_p = 0
            # 角度奖励
            r_a = 1 - (q_r + q_b) / 180
            # 距离奖励
            r_d = 0
            d_max = 1000  # 最大预期距离，在max-min区间内取得最大奖励
            d_min = 200  # 最小预期距离
            if distance < d_min:
                r_d = exp((distance - d_min) / d_min)
            elif d_min <= distance <= d_max:
                r_d = 1
            elif distance > d_max:
                r_d = exp(-(distance - d_max) / d_max)
            # 高度奖励
            dh_opt = 200  # 高于这个500高度差获得最大奖励，低于这个高度差惩罚，越低惩罚越大
            if delta_h < -dh_opt:
                r_h = 0
            elif -dh_opt <= delta_h <= dh_opt:
                r_h = 0.45 + (0.45 * delta_h) / dh_opt
            else:
                r_h = 0.9
            # 速度奖励
            if v_b < 0.6 * v_r:
                r_v = 0.9
            elif (0.6 * v_r) <= v_b <= (1.5 * v_r):
                r_v = 1.5 - v_b / v_r
            else:
                r_v = 0
            # 步数奖励
            # r_s = 0
            r_s = -0.5
            # 获胜奖励
            if (100 <= distance <= 1000) and (q_r <= 30) and (q_b <= 60):  # q_b < 60
                r_end = 20
                script.is_alive = False
                env.win_num += 1
            elif (100 <= distance <= 1000) and (q_r >= 120) and (q_b >= 150):  # q_b > 150
                r_end = -20
                agent.is_alive = False
                env.lose_num += 1
            elif env.steps == self.max_cycles:
                r_end = -20
                agent.is_alive = False
                script.is_alive = False
            else:
                r_end = 0
            # 综合奖励
            w_a = 0.45  # r_a
            w_d = 0.25  # r_d 
            w_h = 0.15  # r_h
            w_v = 0.15  # r_v
            r_inside = w_a * r_a + w_d * r_d + w_h * r_h + w_v * r_v + r_p + r_s
            r_total = r_inside + r_end

            if (env.win_num != last_win_num) or (env.lose_num != last_lose_num):
                env.stop_steps[agent.name] = env.steps
                print("%d round, win_num: %d, lose_num: %d" % (env.steps, env.win_num, env.lose_num))
            return r_total
        else:
            return 0
        
    def observation(self, agent: Agent, env: BaseEnv):
        """
        Observation for close 1v1.
        Only for agents.
        
        NOTE:
        1. ! 只对agent起作用
        """
        world = env.world
        A_MAX = world.constant.A_MAX
        D_MAX = world.constant.D_MAX
        DH_MAX = world.constant.DH_MAX
        DV_MAX = world.constant.DV_MAX
        Z_MAX = world.constant.Z_MAX
        Z_MIN = world.constant.Z_MIN
        V_MAX = world.constant.V_MAX
        V_MIN = world.constant.V_MIN
        state = 0
        r0 = world.agents[0]
        b0 = world.scripts[0]
        x_r, y_r, z_r, phi_r, mu_r, v_r = r0.unpack
        x_b, y_b, z_b, phi_b, mu_b, v_b = b0.unpack
        distance, delta_h, delta_v, q_r, q_b, beta = self.single_situation(r0.unpack, b0.unpack)
        # [q_r, q_b, beta, distance, delta_h, delta_v, mu_r, mu_b, z_r, v_r]
        state = np.array([q_r/A_MAX, q_b/A_MAX, beta/A_MAX, distance/D_MAX, delta_h/DH_MAX, 
                          delta_v/DV_MAX, mu_r/A_MAX, mu_b/A_MAX,
                          (z_r - Z_MIN)/Z_MAX, (v_r - V_MIN)/V_MAX], dtype=np.float32)
        return state
    
    def done(self, agent: Agent, env: BaseEnv):
        """
        Return the observation of single agent.
        """
        success = True
        # 敌方全部消灭或者超出时间
        for script in env.world.scripts:
            if script.is_alive is True:
                success = False

        if success:
            return True
        if agent.is_alive:
            return False
        return True


if __name__ == '__main__':
    env = raw_env()
    test_circle = False

    done_n = [False]
    actions = [2]
    totoal_rewards = np.array([0], dtype=np.float64)

    if test_circle:
        for i in range(10):
            env.reset()
            while False in done_n:
                obs_n, reward_n, done_n, info_n = env.step(actions)
                totoal_rewards += np.array(reward_n)
            print('%d, total_rewards: ' % (i+1) + str(totoal_rewards))
            done_n = [False]
            totoal_rewards = np.array([0], dtype=np.float64)
    else:
        env.reset()
        while False in done_n:
            obs_n, reward_n, done_n, info_n = env.step(actions)
            totoal_rewards += np.array(reward_n)
        print('total_rewards: ' + str(totoal_rewards))
        env.monitor()
        