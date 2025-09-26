import random
from typing import Any

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from envs.mae_parallel.mae_utils.core import Agent, Script, World
from envs.mae_parallel.mae_utils.scenario import BaseScenario

"""
dif from mae:
1. spaces -> space
- space改由list表示 
2. 没有total_observation，observation_space
由各个obs_space append得到
- 没法单独设置total_obs
3. 没有last，step返回last原有的信息
4. seeding from gym(原先是gymnasium)
5. env.stop_condition -> scenario.done
- mae中只能采用env.stop_condition?
6. (暂)不进行agents.remove

关于环境的重新设想:
1. 为core.entity添加is_alive属性，但不添加is_crash、is_shotdown
2. 将step返回的四个变量改为不依赖.self
3. 将done改为通过is_alive判断，在done前先改变is_alive
4. is_alive在reward中判断，因此reward必须设置在done前面
"""

AgentID = str 
ScriptID = str  

class BaseEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "base_name": "mae"
    }
    """
    多机协同空战环境:
    1. 来自mae(based on pettingzoo.mpe)和openai.mpe
    2. 基本框架基于mae，和外部交互的环节改为基于openai.mpe
    3. agents.remove只能通过AgentID进行，不能通过序号
    4. 增加了share_observation_space为了适应marlbenchmark

    和AirCombat_CX相比:
    1. 添加了real_action以适应marlbenmark的离散动作编码方式
    2. 在env_wrappers添加了rews_flag使rews适应算法
    """

    possible_agents: list[AgentID]
    agents: list[AgentID]
    scripts: list[ScriptID]

    def __init__(
        self,
        scenario: BaseScenario,
        world: World,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
        scr_mode='mini_max'
    ):
        super().__init__()  # only pass

        # del render_mode
        self._seed()

        # + all agents get total reward in cooperative case
        self.shared_reward = False

        self.max_cycles = max_cycles  # number of frames (a step for each agent)
        self.scenario = scenario  # 外部scenario
        self.world = world  # 外部world
        self.continuous_actions = continuous_actions  # 是否是连续动作，默认为False
        self.local_ratio = local_ratio  # ?

        self.scenario.reset_world(self.world, self.np_random)  # 初始化位置与速度等

        # 添加agents，设置_agent_selector
        self.agents = [agent.name for agent in self.world.agents]  # agents_ID
        self.possible_agents = list.copy(self.agents)
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }  # {agent.name: idx} 将name对应idx

        # del _agent_selector

        # 添加scripts
        self.scr_on = False
        self.scr_mode = scr_mode
        self.scripts = list()
        if len(self.world.scripts) != 0:
            self.scr_on = True
        if self.scr_on is True:
            self.scripts = [script.name for script in self.world.scripts]
            self.possible_scripts = list.copy(self.scripts)
            self._scripts_map = {
                script.name: idx for idx, script in enumerate(self.world.scripts) 
            }
            # TODO: 添加一个目标分配算法
            self.scripts_to_agents = self.target_assignment()
            self.agents_to_scripts = dict()
            for key in self.scripts_to_agents:
                item = self.scripts_to_agents[key]
                self.agents_to_scripts[item] = key 

        # set spaces
        # spaces -> space, dict -> list
        self.action_space = list()
        self.observation_space = list()
        self.share_observation_space = list()
        share_obs_dim = 0
        # del state_dim
        for agent in self.world.agents:
            if self.continuous_actions:
                space_dim = self.world.dim_n
            else:
                assert self.continuous_actions is False, \
                    'Continuous_actions must be True or False.'
                space_dim = self.world.dim_l

            obs_dim = len(self.scenario.observation(agent, self))
            share_obs_dim += obs_dim

            # action_space，是mae中的spaces
            if self.continuous_actions:
                self.action_space.append(spaces.Box(
                    low=-1, high=1, shape=(space_dim,)
                ))
            else:
                self.action_space.append(spaces.Discrete(space_dim))

            # observation_space，是mae中的spaces
            self.observation_space.append(spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            ))

        self.share_observation_space = [spaces.Box(
        low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.num_agents)]
        
        self.steps = 0
        self.win_num = 0
        self.lose_num = 0
        self.stop_steps = {name: self.max_cycles for name in self.agents}

        # 存储array
        count = self.max_cycles + 1  # 0存储的是还没动作的序列
        num_agents = self.num_agents
        num_scripts = self.num_scripts
        # 6: [x, y, z, phi, mu, gamma]
        self.agents_array = np.zeros((count, num_agents, 6), dtype=float)
        # 没有设置scripts时，scripts_array -> []
        self.scripts_array = np.zeros((count, num_scripts, 6), dtype=float)

        # del state_array

        # 3: 连续动作，1: 离散动作
        self.ag_actions_array = np.zeros((count, num_agents, 3), dtype=float) \
                                if self.continuous_actions is True else \
                                np.zeros((count, num_agents, 1), dtype=float)
        # 1: 离散动作
        self.sc_actions_array = np.zeros((count, num_scripts, 1), dtype=float)

    @property
    def num_agents(self) -> int:
        return len(self.agents)
    
    @property
    def num_scripts(self) -> int:
        return len(self.scripts)

    def _seed(self, seed=None):
        """
        随机数种子设置
        """
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        """
        Reset.
        """
        if seed is not None:
            self._seed(seed=seed)

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = list.copy(self.possible_agents)

        self.steps = 0
        self.win_num = 0
        self.lose_num = 0
        self.stop_steps = {name: self.max_cycles for name in self.agents}

        if self.scr_on is True:
            self.scripts = list.copy(self.possible_scripts)

        # array
        count = self.max_cycles
        num_agents = self.num_agents
        num_scripts = self.num_scripts
        self.agents_array = np.zeros((count, num_agents, 6), dtype=float)
        self.scripts_array = np.zeros((count, num_scripts, 6), dtype=float)
        self.ag_actions_array = np.zeros((count, num_agents, 3), dtype=float) \
                                if self.continuous_actions is True else \
                                np.zeros((count, num_agents, 1), dtype=float)
        self.sc_actions_array = np.zeros((count, num_scripts, 1), dtype=float)
        # return
        obs_n = []  # 不同于self.obs_n
        for a in self.agents:
            obs_n.append(self._get_obs(a))
        return obs_n

    def _execute_world_step(self, action_n):
        """
        Set action for each agent.

        步骤:
        1. 无论是否is_alive，均设置动作(不会出现action=None的情况)
                |
        2. is_alive=True，才存储信息
                |
        3. world.step()，但在world，只有agent is_alive=True，才会实际执行动作
        """
        real_action_n = list()
        for i, agent in enumerate(self.world.agents):
            action = action_n[i]  # 单个agent的动作
            scenario_action = list()  # scenario动作
            scenario_action.append(action)
            real_action = self._set_action(scenario_action, agent, self.world)
            if real_action is not None:
                real_action_n.append(real_action)
        if len(real_action_n) > 0:
            action_n = real_action_n
        # store agents array
        for a in self.agents:
            i = self._index_map[a]
            agent = self.world.agents[i]
            if agent.is_alive is True:
                simu = agent.unpack
                self.agents_array[self.steps, i] = simu
                self.ag_actions_array[self.steps, i] = action_n[i]  # TODO: 这句话要生效，必须ag_actions_array中的i和agent_idx一一对应

        if self.scr_on is True:
            sc_actions = self.scripts_policy()
            for i, script in enumerate(self.world.scripts):
                internal_action = list()
                internal_action.append(sc_actions[i])
                self._set_script_action(internal_action, script)
            # store scripts array
            for s in self.scripts:
                i = self._scripts_map[s]
                script = self.world.scripts[i]
                if script.is_alive is True:
                    simu = script.unpack
                    self.scripts_array[self.steps, i] = simu
                    self.sc_actions_array[self.steps, i] = sc_actions[i]  # 同ag_actions_array

        self.world.step()  # 更新agents.state

        # del reward相关的内容

    def _set_action(self, action, agent: Agent, world: World):
        """
        Set env action for a particular agent.
        """
        agent.action.u = np.zeros(3)

        dt = world.dt
        D2R = np.pi/180  # deg2rad
        R2D = 180/np.pi  # rad2deg
        NX_SCALE = world.constant.NX_SCALE
        NY_SCALE = world.constant.NY_SCALE
        GA_SCALE = world.constant.GA_SCALE

        real_action = None
        if action is not None:
            if self.continuous_actions:
                gamma = agent.action.u[2] if agent.action.u[2] is not None else 0
                gamma_ = float(GA_SCALE * np.clip(action[0][2], -1, 1))
                roll_rate = (gamma_ - gamma) / dt
                if -230 * D2R < roll_rate < 230 * D2R:
                    gamma = gamma_
                elif roll_rate > 230 * D2R:
                    gamma = gamma + (230 * D2R) * dt
                elif roll_rate < -230 * D2R:
                    gamma = gamma - (230 * D2R) * dt
                # rescale red_action
                nx = float(NX_SCALE * np.clip(action[0][0], -1, 1))
                ny = float(NY_SCALE * np.clip(action[0][1], -1, 1))
                agent.action.u[0] = nx
                agent.action.u[1] = ny
                agent.action.u[2] = gamma
            else:
                # process discrete action
                # 'nx_s' = [0, 2, 0, -2, 0, 0, 0]
                # 'ny_s' = [5, 1, 1, 1, 5, 5, -5]
                # 'gamma_s' = [-np.pi/3, 0, 0, 0, np.pi/3, 0, 0]
                if type(action[0]) is int:
                    real_action = int(action[0])
                elif type(action[0]) is np.int32:
                    real_action = int(action[0])
                elif type(action[0]) is np.int64:
                    real_action = int(action[0])
                else:
                    assert len(action[0]) > 1, 'Discrete action must be one-hot or int.'
                    real_action = np.argmax(action[0])
                if real_action == 0:
                    agent.action.u[0] = 0
                    agent.action.u[1] = 5
                    agent.action.u[2] = -np.pi/3
                if real_action == 1:
                    agent.action.u[0] = 2
                    agent.action.u[1] = 1
                    agent.action.u[2] = 0
                if real_action == 2:
                    agent.action.u[0] = 0
                    agent.action.u[1] = 1
                    agent.action.u[2] = 0
                if real_action == 3:
                    agent.action.u[0] = -2
                    agent.action.u[1] = 1
                    agent.action.u[2] = 0
                if real_action == 4:
                    agent.action.u[0] = 0
                    agent.action.u[1] = 5
                    agent.action.u[2] = np.pi/3
                if real_action == 5:
                    agent.action.u[0] = 0
                    agent.action.u[1] = 5
                    agent.action.u[2] = 0
                if real_action == 6:
                    agent.action.u[0] = 0
                    agent.action.u[1] = -5
                    agent.action.u[2] = 0
            # 这个还有使用完action的判断功能
            action = action[1:]  # 如果有action.c，那就是action.c
            # make sure we used all elements of action
            assert len(action) == 0
            return real_action

    def _set_script_action(self, action, script: Script):
        """
        Set env action for a particular script.
        Only discrete action.
        """
        script.action.u = np.zeros(3)

        if action is not None:
            # process discrete action
            # TODO: 这里是否要从json读入
            # 'nx_s' = [0, 2, 0, -2, 0, 0, 0]
            # 'ny_s' = [5, 1, 1, 1, 5, 5, -5]
            # 'gamma_s' = [-np.pi/3, 0, 0, 0, np.pi/3, 0, 0]
            if action[0] == 0:
                script.action.u[0] = 0
                script.action.u[1] = 5
                script.action.u[2] = -np.pi/3
            if action[0] == 1:
                script.action.u[0] = 2
                script.action.u[1] = 1
                script.action.u[2] = 0
            if action[0] == 2:
                script.action.u[0] = 0
                script.action.u[1] = 1
                script.action.u[2] = 0
            if action[0] == 3:
                script.action.u[0] = -2
                script.action.u[1] = 1
                script.action.u[2] = 0
            if action[0] == 4:
                script.action.u[0] = 0
                script.action.u[1] = 5
                script.action.u[2] = np.pi/3
            if action[0] == 5:
                script.action.u[0] = 0
                script.action.u[1] = 5
                script.action.u[2] = 0
            if action[0] == 6:
                # 以前的env，这里根据phi的不同，会有不同的参数
                # 但是这里script.action.u[0] = 0
                script.action.u[0] = 0
                script.action.u[1] = -5
                script.action.u[2] = 0
            # 这个还有使用完action的判断功能
            action = action[1:]  # 如果有action.c，那就是action.c
            # make sure we used all elements of action
            assert len(action) == 0
    
    def _get_info(self, agent: AgentID):
        """
        Get info used for benchmarking.
        """
        return self.scenario.benchmark_data(
            self.world.agents[self._index_map[agent]], self
        )

    def _get_obs(self, agent: AgentID):
        """
        Get observation for a particular agent.
        """
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self
        ).astype(np.float32)

    def _get_done(self, agent: AgentID):
        """
        Get dones for a particular agent.
        """
        return self.scenario.done(
            self.world.agents[self._index_map[agent]], self
        )

    def _get_reward(self, agent: AgentID):
        """
        Get reward for a particular agent.
        """
        return self.scenario.reward(
            self.world.agents[self._index_map[agent]], self
        )

    def _get_global_reward(self):
        """
        Get global_reward for all agents.
        """
        return self.scenario.global_reward(self)

    def step(self, action_n: list):
        """
        Step(like openai.mpe).

        步骤:
        1. 再次赋值agents(可能没有意义)
                |
        2. 将外部action一起传入，然后通过循环进行set_action
                |
        3. 进行world.step()
                |
        4. 获取obs_n, reward_n, done_n, info_n
        """
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        self._execute_world_step(action_n)
        self.steps += 1

        for a in self.agents:
            obs_n.append(self._get_obs(a))
            reward_n.append(self._get_reward(a))
        
        global_reward, win = self._get_global_reward()
        for i, r in enumerate(reward_n):
            r += global_reward
            reward_n[i] = r
        if win != 2:
            info_n['w'] = win

        for a in self.agents:
            done_n.append(self._get_done(a))
            info_n['n'].append(self._get_info(a))

        # all agents get total reward in cooperative case
        if self.shared_reward:
            reward = np.mean(reward_n)  # 将sum改为mean
            reward_n = [reward] * self.num_agents  # 更改

        return obs_n, reward_n, done_n, info_n

    def target_assignment(self) -> dict[ScriptID, AgentID]:
        """
        The target_assignment for scripts.
        num_agents >= num_scripts.
        
        attr:
        1. self.agents: list(agent.name)
        2. self.scripts: list(script.name)
        3. self._index_map: {agent.name: idx(from world.agents)}
        4. self._scripts_map: {script.name: idx(from world.scripts)}
        5. self.scripts_to_agents: {script.name: agent.name}
        """
        # agents >= scripts
        # 简单的距离最近进行分配
        current_agents = list.copy(self.agents)
        goal_a = current_agents[0]
        scripts_to_agents = dict()
        dis = 0
        for s in self.scripts:
            script_idx = self._scripts_map[s]
            for a in current_agents:
                agent_idx = self._index_map[a]
                dis_ = self.scenario.single_situation(
                        self.world.scripts[script_idx].unpack,
                        self.world.agents[agent_idx].unpack)[0]
                if dis_ > dis:
                    goal_a = a
                dis = dis_
            scripts_to_agents[s] = goal_a
            current_agents.remove(goal_a)
        return scripts_to_agents

    def scripts_policy(self):
        """
        The policy for all scripts.
        This method should not be called when 
        self.scr_on is not True.

        single mini_max(blue):
        - TODO: 这和真正的mini_max并不一致，或者说叫做max_max
        - TODO: 将script_action: None放在这里设置
        - 因为red获得最大奖励，并不意味blue获得奖励最小
        - 但在很多情况下两者应该是几乎等价的
        1. 挑选出red让red能获取最大奖励的动作(blue没动)
        - 这个动作执行和实际动作的执行顺序并不一致
                |
        2. 得到该red动作下的red_simu_
                |
        3. 在red_simu_的基础上，挑选出blue能获取最大奖励的动作
        """
        # scripts必须存在，否则该方法不应该被调用
        assert self.scr_on is True, \
            'This method should not be called.'

        # 采取len(self._scripts_map)，因为self._scripts_map不会变化
        sc_actions: list[Any]  # TODO: 为了解决警告
        sc_actions = [None] * len(self._scripts_map)

        assert self.scr_mode is not None, 'Must get scr_mode.'

        if self.scr_mode == 'steady':
            # 即使是固定动作，也根据self.scripts赋值
            for s in self.scripts:
                script_idx = self._scripts_map[s]
                sc_actions[script_idx] = 2
        if self.scr_mode == 'random':
            for s in self.scripts:
                script_idx = self._scripts_map[s]
                sc_actions[script_idx] = random.randint(0, 6)
        if self.scr_mode == 'mini_max':
            for s in self.scripts:
                a = self.scripts_to_agents[s]
                agent_idx = self._index_map[a]
                script_idx = self._scripts_map[s]
                agent = self.world.agents[agent_idx]
                script = self.world.scripts[script_idx]
                action_b = self.scenario.mini_max(script, agent, self.world, self.steps)
                sc_actions[script_idx] = action_b
        return sc_actions
    
    def agents_by_script(self):
        """
        Only for debug.
        """
        assert self.scr_on is True, \
            'This method should not be called.'
        ag_actions: list[Any]
        ag_actions = [None] * len(self._index_map)
        for a in self.agents:
            s = self.agents_to_scripts[a]
            agent_idx = self._index_map[a]
            script_idx = self._scripts_map[s]
            agent = self.world.agents[agent_idx]
            script = self.world.scripts[script_idx]
            action_r = self.scenario.mini_max(agent, script, self.world, self.steps)
            ag_actions[script_idx] = action_r
        return ag_actions

    def monitor(self):  
        """
        The monitor of env.
        """
        pass
