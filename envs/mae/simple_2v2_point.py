import numpy as np
from gymnasium.utils import EzPickle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO

from envs.mae.mae_utils.core import Agent, Script, World
from envs.mae.simple_2v2 import CloseConstants
from envs.mae.mae_utils.scenario import BaseScenario
from envs.mae.mae_utils.base_env import BaseEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

"""
    simple 2v2:
    1. red: agent, steady
    2. blue: script, mini_max

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
    def __init__(self, N=2, max_cycles=200, continuous_actions=False, 
                 render_mode=None, scr_mode='mini_max', shared_reward=False,
                 tar_mode=2):
        EzPickle.__init__(  # 注意新增输入参数要在此添加
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            scr_mode=scr_mode,
            shared_reward=shared_reward,
            tar_mode=tar_mode
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
            scr_mode=scr_mode,
            shared_reward=shared_reward,
            tar_mode=tar_mode
        )  # 将设置好的world等传入BaseEnv基类中进行初始化
        self.metadata["name"] = "simple_2v2"

    def monitor(self):
        # TODO: 代码冗余过多，在之后将循环合并
        count = self.max_cycles + 1

        # Maneuver Trajectory
        fig = plt.figure()
        # ax1 = fig.plot(projection='3d')
        ax1 = Axes3D(fig)  # 只会在matplotlib 3.5.1中生效
        # 这样只画出了scripts_to_agents中的entity
        for s in self.possible_scripts:
            a = self.scripts_to_agents[s]
            agent_idx = self._index_map[a]
            script_idx = self._scripts_map[s]

            red_array = self.agents_array[:, agent_idx, :]
            blue_array = self.scripts_array[:, script_idx, :]
            step_num = self.stop_steps[a]

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
        for s in self.possible_scripts:
            a = self.scripts_to_agents[s]
            agent_idx = self._index_map[a]
            script_idx = self._scripts_map[s]

            red_array = self.agents_array[:, agent_idx, :]
            blue_array = self.scripts_array[:, script_idx, :]
            step_num = self.stop_steps[a]

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

        # TODO: 添加obs


# 包装
env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    """
    设置world的具体脚本:
    1. 通过make_world初始化一个world
    2. 通过reset_world进行位置和速度矢量初始化

    NOTE:
    1. reset_world是必须的，因为make_world中没有初始化位置与速度
    - TODO: 这点是否和dead_agent相关
    """

    def __init__(self, N=2, max_cycles=200):
        """
        将N、max_cycles作为实例化时传入
        """
        super().__init__()
        self.N = N
        self.max_cycles = max_cycles
        self.single_global = False  # single reward for global
        self.single_each = True  # single reward for global

    def make_world(self):
        """
        make_world完成了以下:
        1. 设置agents、scripts的个数
        2. 为agent添加了name
        3. 为script添加了name
        """
        world = World(CloseConstants())  # 实例化world
        # set any world properties first
        num_agents = self.N
        world.num_agents = num_agents
        num_script = num_agents
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

        # +, 更改的话需要再reset_world中添加
        for agent in world.agents:
           agent.r_d_before = 0
           agent.max_r_d = 0
           agent.gain = 0  # 击败敌机数目
        return world

    def reset_world(self, world: World, np_random):
        """
        reset_world完成了以下:
        1. 初始化agents、scripts的位置和速度
            ！这里不随机化
        2. TODO: 这里的np_random for seed是否生效未知

        NOTE:
        1. TODO: 位置还要重新修改
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
        # red_1
        r1 = world.agents[1]
        r1.initial_state(np.array([1000, 0, 3000, 45, 0, 200]))
        # blue_0
        b0 = world.scripts[0]
        b0.initial_state(np.array([3000, 3000, 3000, 90, 0, 220]))
        # blue_1
        b1 = world.scripts[1]
        b1.initial_state(np.array([4000, 4000, 3000, 90, 0, 220]))

        # +, 更改的话需要再reset_world中添加
        for agent in world.agents:
           agent.r_d_before = 0
           agent.max_r_d = 0
           agent.gain = 0  # 击败敌机数目
    
    def single_decide(self, agent: Agent, script: Script):
        if (agent.is_alive and script.is_alive) is False:
            return 0
        return self.single_close_reward(agent.unpack, script.unpack, end_on=False)

    def global_reward(self, env: BaseEnv):
        """
        现在的逻辑:
        1. 不因循环数而添加r_global
        - 因为可能会与胜负全局奖励重合
        2. 平局的全局奖励改为0

        NOTE:
        1. 现在agent dead还是会获得全局奖励
        2. TODO: 为了获得超出时间限制的惩罚，将其转移至reward
        3. TODO: active_masks后，dead agents的reward是否会起作用
        - 如果不起作用，那么获胜奖励将会作废(如果agents和scripts同步dead)
        """
        r_global = 0
        world = env.world
        # single_reward
        if self.single_global:
            single_reward_array = list()
            single_reward_array.append(self.single_decide(world.agents[0], world.scripts[0]) + \
                                       self.single_decide(world.agents[1], world.scripts[1]))
            single_reward_array.append(self.single_decide(world.agents[0], world.scripts[1]) + \
                                       self.single_decide(world.agents[1], world.scripts[0]))
            if 0 in single_reward_array:
                single_reward_array.remove(0)
            r_global += np.max(single_reward_array) / 2

        agents_dead = True
        scripts_dead = True
        for agent in env.world.agents:
            if agent.is_alive is True:
                agents_dead = False
        for script in env.world.scripts:
            if script.is_alive is True:
                scripts_dead = False

        win = 2
        if agents_dead or scripts_dead:  # 包括了(1, 1), (2, 0), (0, 2)
            if env.win_num > env.lose_num:
                # r_global += 20
                win = 1
            elif env.win_num < env.lose_num:
                # r_global -= 20
                win = -1
            elif env.win_num == env.lose_num:
                # r_global = 0
                win = 0
        return r_global, win

    def reward(self, agent: Agent, env: BaseEnv):
        """
        Reward for close 2v2.

        NOTE:
        1. 将输入参数中的agent改为entity
        2. agent, script必须分开获得reward
        3. 输入参数的形式不好改变
        4. entity, world中都不含步数信息，所有还需要一个步数作为输入参数
        5. 通过distances，可以避免由于inf带来的错误
        """
        world = env.world
        D_MAX = world.constant.D_MAX
        distances = []

        last_win_num = env.win_num
        last_lose_num = env.lose_num
        r_total = 0
        r_inside = 0
        r_end = 0
        if agent.is_alive is False:
            return 0
        for s in env.scripts:
            s_index = env._scripts_map[s]
            script = env.world.scripts[s_index]

            if script.is_alive is False:
                continue

            distance, delta_h, delta_v, q_r, q_b, beta_rb = \
                self.single_situation(agent.unpack, script.unpack)
            
            distances.append(distance)

            if (100 <= distance <= 1000) and (q_r <= 30) and (q_b <= 60):  # q_b < 60
                r_end = 10
                # agent.is_alive = False
                script.is_alive = False
                env.win_num += 1
                agent.gain += 1
                break
            elif (100 <= distance <= 1000) and (q_r >= 120) and (q_b >= 150):  # q_b > 150
                r_end = -10
                agent.is_alive = False
                script.is_alive = False
                env.lose_num += 1
                agent.gain -= 1
                break
            elif (env.steps == self.max_cycles) and (agent.gain <= 0):
                r_end = -10
                agent.is_alive = False
                script.is_alive = False
                break
            else:
                r_end = 0    

        if self.single_each:  # 和single_global相冲突
            r_s = 0
            if len(distances) > 0:
                r_d_current = - min(distances) / D_MAX * 50
                if agent.r_d_before == 0:
                    r_d = 0
                else:
                    r_d = r_d_current - agent.r_d_before
                    if 100 <= min(distances) <= 1000:
                        # r_d = agent.max_r_d
                        r_d = 0.2
                agent.r_d_before = r_d_current
            else:
                r_d_current = 0
                r_d = r_d_current
            if r_d > agent.max_r_d:
                agent.max_r_d = r_d
            r_s = -0.2
            r_inside = r_d + r_s

        r_total = r_inside + r_end

        if (env.win_num != last_win_num) or (env.lose_num != last_lose_num):
            env.stop_steps[agent.name] = env.steps
            print("%d round, win_num: %d, lose_num: %d" % (env.steps, env.win_num, env.lose_num))
        return r_total

    def observation(self, agent: Agent, env: BaseEnv):
        """
        Observation for close 1v1.
        Only for agents.
        
        NOTE:
        1. ! 只对agent起作用
        2. q_r, q_b, beta, distance, delta_h, delta_v, 
        mu_r, mu_b, z_r, v_r:
        - mu_r, z_r, v_r: agent本身属性
        - mu_b: other属性
        - q_r, q_b, beta, distance, delta_h, delta_v: situation得到
        3. TODO: 目前的delta_v归一化后似乎过小
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
        state = []

        x_r, y_r, z_r, phi_r, mu_r, v_r = agent.unpack
        state.append([mu_r/A_MAX, (z_r - Z_MIN)/Z_MAX, (v_r - V_MIN)/V_MAX])
        for s in env.possible_scripts:
            script = world.scripts[env._scripts_map[s]]
            # if script.is_alive:
            #     x_b, y_b, z_b, phi_b, mu_b, v_b = script.unpack
            #     distance, delta_h, delta_v, q_r, q_b, beta = self.single_situation(agent.unpack, script.unpack)
            #     state.append([mu_b/A_MAX, q_r/A_MAX, q_b/A_MAX, beta/A_MAX, distance/D_MAX, delta_h/DH_MAX, 
            #                   delta_v/DV_MAX])
            # else:
            #     state.append([0, 0, 0, 0, 0, 0, 0])
            x_b, y_b, z_b, phi_b, mu_b, v_b = script.unpack
            distance, delta_h, delta_v, q_r, q_b, beta = self.single_situation(agent.unpack, script.unpack)
            state.append([mu_b/A_MAX, q_r/A_MAX, q_b/A_MAX, beta/A_MAX, distance/D_MAX, delta_h/DH_MAX, 
                            delta_v/DV_MAX])
        state = np.concatenate(state, dtype=np.float32)
        return state
    
    def done(self, agent: Agent, env: BaseEnv):
        """
        Return done of single agent for terminations.
        """
        stop_flag = True  # 敌方全部消灭或者超出时间
        for script in env.world.scripts:
            if script.is_alive is True:
                stop_flag = False

        if stop_flag:
            return True
        if agent.is_alive:
            return False
        return True
    

if __name__ == '__main__':
    from pettingzoo.test import api_test
    test_api = False  # 会提示space过大过小
    ac_mode = 1
    env = raw_env(scr_mode='mini_max')
    env.reset()
    
    if test_api is True:
        api_test(env, num_cycles=10, verbose_progress=True)  # num_cycles超出赢的步数会报错
    else:
        reward_array = np.zeros((201, 2, 1), dtype=float)
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            reward_array[env.steps, env._index_map[agent]] = reward
            if ac_mode == 0:
                action = 2  # 从space采样就是int
            elif ac_mode == 1:
                action = env.action_space(agent).sample()
            if termination is True or truncation is True:
                action = None
            env.step(action)
        env.monitor()
        