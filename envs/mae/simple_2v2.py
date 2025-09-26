import numpy as np
from gymnasium.utils import EzPickle
from typing import Union, Optional, List

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from envs.mae.mae_utils.core import Agent, Script, World, ExtraConstants
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
"""

class CloseConstants(ExtraConstants):
    def __init__(self):
        super().__init__()
        # like ExtraConstants


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
    def __init__(self, 
                 N: int =2, 
                 max_cycles: int = 200, 
                 continuous_actions: bool = False, 
                 render_mode: Optional[str] = None, 
                 scr_mode: str = 'mini_max', 
                 shared_reward: bool = False,
                 tar_mode: str = 'auction',
                 simu_init: Union[List, np.ndarray] = None,
                 use_seed: bool = True,
                 selfplay: bool = False):
        EzPickle.__init__(  # 注意新增输入参数要在此添加
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            scr_mode=scr_mode,
            shared_reward=shared_reward,
            tar_mode=tar_mode,
            simu_init=simu_init,
            use_seed=use_seed,
            selfplay=selfplay
        )  # 初始化一些参数
        scenario = Scenario(N, max_cycles, scr_mode, simu_init, use_seed, selfplay)  # 实例化脚本
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

    def monitor(self, 
                en: bool = True, 
                only_fig: bool = False, 
                fig_path: str = None,
                sub_fig: bool = False):
        from envs.mae.sub_utils.figure_utils import traj_plot, obs_plot_0, reward_plot, \
                                                    action_plot
        path = fig_path
        traj_plot(self, en, only_fig, path)
        if en is False:
            obs_plot_0(self, only_fig, path)
            if sub_fig is True:
                reward_plot(self, only_fig, path)
                action_plot(self, only_fig, path)


# 包装
env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):

    def __init__(self, 
                 N: int =2, 
                 max_cycles: int = 200,
                 scr_mode: Optional[str] = None,
                 simu_init: Union[List, np.ndarray] = None,
                 use_seed: bool = True,
                 selfplay: bool = False):
        """
        将N、max_cycles作为实例化时传入
        """
        super().__init__()
        self.N = N
        self.max_cycles = max_cycles
        self.single_global = False  # single reward for global
        self.single_each = True  # single reward for global
        self.scr_mode = scr_mode
        self.selfplay = selfplay  # 是否自我对战
        if simu_init is None:
            self.simu_init = [[0, 0, 3000, 45, 0, 200],
                              [1000, 1000, 3000, 45, 0, 200],
                              [3000, 3000, 3000, 90, 0, 220],
                              [4000, 4000, 3000, 90, 0, 220]]
        elif isinstance(simu_init, (np.ndarray, list)):
            self.simu_init = simu_init
        else:
            raise ValueError("simu_init must be list or ndarray")
        self.use_seed = use_seed
        if selfplay:
            if scr_mode == "ippo":
                from stable_baselines3 import PPO
                b_model_dir = "../../runner_pettingzoo/ppo/sub_model/simple_2v2_ippo.pkl"
                self.b_model = PPO.load(b_model_dir)
            if scr_mode == "isac":
                from stable_baselines3 import SAC
                b_model_dir = "../../runner_pettingzoo/sac/sub_model/simple_2v2_isac.pkl"
                self.b_model = SAC.load(b_model_dir)
            if scr_mode == "iddpg":
                from stable_baselines3 import DDPG
                b_model_dir = "../../runner_pettingzoo/ddpg/sub_model/simple_2v2_iddpg.pkl"
                self.b_model = DDPG.load(b_model_dir)
        else:
            if scr_mode == "ppo":
                from stable_baselines3 import PPO
                b_model_dir = "../../runner_pettingzoo/ppo/sub_model/close_combat_ppo.pkl"
                self.b_model = PPO.load(b_model_dir)

    def b_model_policy(self, script: Script, agent: Agent, env: BaseEnv):
        """
        自博弈训练时需要统一导入SB3模型与任务观测的维度
        SARL系统ppo观测为10维，MARL系统观测为17维
        """
        world = env.world
        A_MAX = world.constant.A_MAX
        D_MAX = world.constant.D_MAX
        DH_MAX = world.constant.DH_MAX
        DV_MAX = world.constant.DV_MAX
        Z_MIN = world.constant.Z_MIN
        Z_MAX = world.constant.Z_MAX
        V_MIN = world.constant.V_MIN
        V_MAX = world.constant.V_MAX
        state = []
        assert self.b_model is not None, 'This method should not be called.'
        if self.selfplay:
            x_r, y_r, z_r, phi_r, mu_r, v_r = script.unpack
            state.append([mu_r/A_MAX, (z_r - Z_MIN)/Z_MAX, (v_r - V_MIN)/V_MAX])
            for a in env.possible_agents:
                agent = world.agents[env._index_map[a]]
                x_b, y_b, z_b, phi_b, mu_b, v_b = agent.unpack
                distance, delta_h, delta_v, q_r, q_b, beta = self.single_situation(agent.unpack, script.unpack)
                state.append([mu_b/A_MAX, q_r/A_MAX, q_b/A_MAX, beta/A_MAX, distance/D_MAX, delta_h/DH_MAX, 
                                delta_v/DV_MAX])
            state = np.concatenate(state, dtype=np.float32)
        else:
            distance, delta_h, delta_v, q_r, q_b, beta = self.single_situation(agent.unpack, script.unpack)
            x_r, y_r, z_r, phi_r, mu_r, v_r = agent.unpack
            x_b, y_b, z_b, phi_b, mu_b, v_b = script.unpack
            q_b = 180 - q_b
            q_r = 180 - q_r
            delta_h = -delta_h
            delta_v = -delta_v
            # [q_b, q_r, beta, distance, delta_h, delta_v, mu_b, mu_r, z_b, v_b]
            state = np.array([q_b/A_MAX, q_r/A_MAX, beta/A_MAX, distance/D_MAX, delta_h/DH_MAX, 
                            delta_v/DV_MAX, mu_b/A_MAX, mu_r/A_MAX,
                            (z_b - Z_MIN)/Z_MAX, (v_b - V_MIN)/V_MAX], dtype=np.float32)
        action_b, _ = self.b_model.predict(observation=state)
        return action_b

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
        return world

    def seed_state(self, np_random):
        LOC_SEED = 200
        OTH_SEED = 1
        simu_r0 = np.copy(self.simu_init[0]).astype(np.float32)
        simu_r1 = np.copy(self.simu_init[1]).astype(np.float32)
        simu_b0 = np.copy(self.simu_init[2]).astype(np.float32)
        simu_b1 = np.copy(self.simu_init[3]).astype(np.float32)
        dloc_array = np_random.uniform(-LOC_SEED, LOC_SEED, 3 * 4)
        doth_array = np_random.uniform(-OTH_SEED, OTH_SEED, 3 * 4)
        # r0，注意这种方法只能用在np.array上
        simu_r0[0:3] += dloc_array[0:3]
        simu_r0[3:6] += doth_array[0:3]
        # r1
        simu_r1[0:3] += dloc_array[3:6]
        simu_r1[3:6] += doth_array[3:6]
        # b0
        simu_b0[0:3] += dloc_array[6:9]
        simu_b0[3:6] += doth_array[6:9]
        # b1
        simu_b1[0:3] += dloc_array[9:12]
        simu_b1[3:6] += doth_array[9:12]
        if self.scr_mode == 'steady':
            simu_b0[4] = 0  # mu
            simu_b1[4] = 0  # mu
        simu_seed = np.array([simu_r0, simu_r1, simu_b0, simu_b1])
        return simu_seed

    def reset_world(self, world: World, np_random):
        """
        reset_world完成了以下:
        1. 初始化agents、scripts的位置和速度
            ！这里不随机化
        atr:
        1. fig_step: 绘图需要的step
        2. stop_steps: 环境交互停止的step
        """
        # set is_alive
        for agent in world.agents:
            agent.is_alive = True
            agent.action.u = None
            agent.fig_step = self.max_cycles
            agent.r_d_before = 0
            agent.max_r_d = 0
            agent.gain = 0  # 击败敌机数目
            agent.gain_names = list()
        for script in world.scripts:
            script.is_alive = True
            script.action.u = None
            script.fig_step = self.max_cycles
        # set initial states
        simu_seed = self.simu_init
        if self.use_seed:
            simu_seed = self.seed_state(np_random)
        # only for test
        # temp = simu_seed.astype(np.int32)
        # 应当减少不必要的对象引用
        world.agents[0].initial_state(simu_seed[0])
        world.agents[1].initial_state(simu_seed[1])
        world.scripts[0].initial_state(simu_seed[2])
        world.scripts[1].initial_state(simu_seed[3])
    
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
        2. TODO: active_masks后，dead agents的reward是否会起作用
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
        delta_h_s = []

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
            delta_h_s.append(delta_h)

            if (100 <= distance <= 1000) and (q_r <= 30) and (q_b <= 60):  # q_b < 60
                r_end = 10
                # agent.is_alive = False
                script.is_alive = False
                env.win_num += 1
                script.fig_step = env.steps
                # agent.fig_step = env.steps
                agent.gain += 1
                agent.gain_names.append(script.name)
                break
            elif (100 <= distance <= 1000) and (q_r >= 120) and (q_b >= 150):  # q_b > 150
                r_end = -10
                agent.is_alive = False
                script.is_alive = False
                env.lose_num += 1
                script.fig_step = env.steps
                agent.fig_step = env.steps
                agent.gain -= 1
                break
            elif (env.steps == self.max_cycles) and (agent.gain <= 0):
                r_end = -10
                agent.is_alive = False
                script.is_alive = False
                break
            else:
                r_end = 0    

        if env.win_num == 2:
            for a in env.agents:
                a_index = env._index_map[a]
                agent = env.world.agents[a_index]
                # 没有被敌机击败，但是敌机已全部被击败
                if agent.fig_step == self.max_cycles:
                    agent.fig_step = env.steps
                

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
                    elif min(distances) < 100:
                        r_d = -1
                agent.r_d_before = r_d_current
            else:
                r_d_current = 0
                r_d = r_d_current
            if r_d > agent.max_r_d:
                agent.max_r_d = r_d
            r_s = -0.2
            r_inside = r_d + r_s

        # 高度惩罚
        if agent.unpack[2] < 1000:
            r_h = -1
        else:
            r_h = 0

        # 高度差惩罚
        if len(delta_h_s) > 0 and max(delta_h_s) < 0:
            r_h_d = -0.1
        else:
            r_h_d = 0

        r_total = r_inside + r_end + r_h + r_h_d

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
    import os, sys  
    os.chdir(sys.path[0])
    test_api = False  # 会提示space过大过小
    ac_mode = "mini_max"
    scr_mode = "ppo"
    simu_init = [[4000, 4000, 3000, 90, 0, 220],  # 为了测试b_model_policy的准确性
                 [3500, 3500, 3000, 90, 0, 220],
                 [500, 500, 3000, 45, 0, 200],
                 [1000, 1000, 3000, 45, 0, 200]]
                # [[3000, 3000, 3000, 90, 0, 220],
                #  [4000, 4000, 3000, 90, 0, 220],
                #  [0, 0, 3000, 45, 0, 200],
                #  [1000, 1000, 3000, 45, 0, 200]]
                
    env = raw_env(scr_mode=scr_mode, tar_mode='auction', simu_init=simu_init)
    env.reset(seed=123)
    
    if test_api is True:
        api_test(env, num_cycles=10, verbose_progress=True)  # num_cycles超出赢的步数会报错
    else:
        reward_array = np.zeros((201, 2, 1), dtype=float)
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            reward_array[env.steps, env._index_map[agent]] = reward
            if ac_mode == "steady":
                action = 2  # 从space采样就是int
            elif ac_mode == "random":
                action = env.action_space(agent).sample()
            elif ac_mode == "mini_max":
                script = env.agents_to_scripts[agent]
                action = env.scenario.mini_max(env.world.agents[env._index_map[agent]], 
                                               env.world.scripts[env._scripts_map[script]],
                                               env)
            if termination or truncation:  # 这里的终止条件和sb3.ippo不相同
                action = None
            env.step(action)
        # env.monitor()
        