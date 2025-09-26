import random
from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from envs.mae.mae_utils.core import Agent, Script, World
from envs.mae.mae_utils.scenario import BaseScenario
from envs.mae.sub_utils.auction_for_mae import Auction

from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

"""
    关于ad_env循环:
    1. 获取obs
        |
    2. red获取动作
        |
    3. blue获取动作
        |
    4. red实施动作(只是动力学方程，这里未获取obs)
        |
    5. blue实施动作(只是动力学方程，这里未获取obs)

    关于SimpleEnv循环:
    (scripts和agents必须为敌对关系)
    1. 获取obs(单个entity的信息), state(全局env的信息)
        |
    2. agents获取动作
        |
    3. scripts获取动作(scripts存在才调用)
    - scripts获取动作不需要经过observe_space, state_space
        |
    4. agents实施动作(只是动力学方程，这里未获取obs/state)
        | 3 发生调用
    5. scripts实施动作(只是动力学方程，这里未获取obs/state)

    关于agents_select:
    1. 外部scenario设置world.agents和agent.name
        |
    2. 设置self.agents，里面放置agent.name;
        设置self._index_map: {agent.name: idx}
        |
    3. 将self.agents放入pettingzoo.utils.agent_selector
        以设置self._agent_selector -> next() ->
        self.agent_selection(agent.name)
        |
    4. 依靠_agent_selector，
        agent_selection、_index_map进行配合来完成一个周期的动作循环
        |
    5. 一个agents动作周期循环完后，再设置sc_actions
    - sc_actions目前放在_execute_world_step()中
    - 不一定需要将sc_actions放入step()，因为不需要循环scripts的action
    - 该循环要正确停止，必须对step()进行覆写

    关于script_select:
    1. 假设num_agents >= num_scripts，script_dead意味着也有agent_dead
        |
    2. 首先进行目标分配，每个script分配一个agent -> dict[ScriptID: AgentID]
        |
    3. env.step() -> if next_idx == 0 -> scripts集体进行动作
        |
    4. 只让活着的agent对应的script进行动作

    scenario.:
    1. reset_world
    2. observation
    3. total_observation
    4. global_reward
    5. reward
"""

AgentID = str  
ScriptID = str  

def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class BaseEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }
    """
    多机协同空战环境的假定:
    1. red(agents): n v blue(agents): n，两方的数量均等
    2. red[i]被击败，那么blue[i]也应该退出
    3. scripts如果加入作战，那么和agents必须是敌对方
    4. scripts在开始的时候即进行目标分配，每个scripts分配一个目标，
        后面将不再进行
    5. red(agents): n  >= blue(scripts): n
    6. script_dead意味着也有agent_dead

    多机协同空战环境的特性:
    1. agent_dead or script_dead只是意味着不再接受动作，
        这个entity本身还是存在着，只是entity.state不在变化
    2. 外部通过for agent in env.agent_iter()，每次只传入一个agent的动作
    3. 如果有agent停止了，在外部将这个agent.action设为None
    4. 在dead相关函数调用外，使用list.copy(self.agents)以调用self.agents
    5. ! agent指代Agent，a指代AgentID，但由于继承关系，在不存在a的情况下，也可能用agent指代AgentID
    """

    possible_agents: list[AgentID]
    possible_scripts: list[ScriptID]
    agents: list[AgentID]
    scripts: list[ScriptID]

    def __init__(
        self,
        scenario: BaseScenario,
        world: World,
        max_cycles: int,
        render_mode: Optional[str] = None,
        continuous_actions: bool = False,
        local_ratio: Optional[float] = None,
        scr_mode: str = 'mini_max',
        shared_reward: bool = False,
        tar_mode: str = 'auction',
    ):
        """
        attr:
        1. num_agents从AECEnv继承得到; 添加num_scripts(len(scripts) == 0 -> 0)
        2. self.steps指的是周期数(一个周期: 所有entity做完动作)
        3. scr_on在使用scripts相关方法前都需要经过一次判断
        4. 存储array:
        - agents_array: 存储agents的状态
        - scripts_array: 存储agents的状态
        - situations_array: 存储situations
        - (已删除)state_array: 存储env的total_observation
        - ag_actions_array: 存储agents的action
        - sc_actions_array: 存储agents的action
        - ! ad_env在每次step()之前存储一次array
        5. reward:
        - rewards(会del dead): 每个周期所有agent的奖励
        - _clear_rewards: 清空rewards中的值，每个last()前都会进行一次
        - _cumulative_rewards(会del dead): 每个周期所有agent的奖励，
                                在每个周期前都会对归零
        - _accumulate_rewards: 计算每个周期agent的累计奖励的函数

        agents:
        1. self.world.agents: list[core.Agent()]
        2. self.agents: list[AgentID]，由外部scenario设置agent.name
           在SimpleEnv中赋给AgentID
        3. current_actions: step()传入的action，只有传完一遍才会执行动作
        4. agent.name: 在外部scenario设置
        5. a: self.agents中的AgentID
        6. s: self.scripts中的ScriptID

        dims:
        1. state_dim: agents obs_dim的总和,
        - ! 在空战场景中state_dim可能需要在外部scenario重新确定
        2. space_dim: agent的action_space，分为连续和离散
        - 由world.dim_n、world.dim_l决定
        - world.dim_n = 3  连续动作空间维度 [nx, ny, gamma]
        - world.dim_l = 7  离散动作空间维度 机动动作库(library)
        3. obs_dim: agent(good or adversary)的obs维度

        NOTE:
        1. 这里设置了obs_space、state_space、action_space
        2. 实际返回的obs、state在具体函数中设置
        3. 这里的space只是简单地设置为[-inf, inf]
        4. 该类不包含landmark的执行
        """
        super().__init__()  # only pass

        self.render_mode = render_mode  # render_mode现在不起作用
        self._seed()

        # + all agents get total reward in cooperative case
        self.shared_reward = shared_reward

        self.max_cycles = max_cycles  # number of frames (a step for each agent)
        self.scenario = scenario  # 外部scenario
        self.world = world  # 外部world
        self.continuous_actions = continuous_actions  # 是否是连续动作，默认为False
        self.local_ratio = local_ratio  # ?

        self.scenario.reset_world(self.world, self.np_random)  # 这是必须的

        # 添加agents，设置_agent_selector
        self.agents = [agent.name for agent in self.world.agents]  # agents_ID
        self.possible_agents = list.copy(self.agents)
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }  # {agent.name: idx} 将name对应idx

        self._agent_selector = agent_selector(self.agents)

        # 添加scripts
        self.scr_on = False
        self.scr_mode = scr_mode
        self.tar_mode = tar_mode
        self.scripts = list()
        if len(self.world.scripts) != 0:
            self.scr_on = True
        if self.scr_on is True:
            self.scripts = [script.name for script in self.world.scripts]
            self.possible_scripts = list.copy(self.scripts)
            self._scripts_map = {
                script.name: idx for idx, script in enumerate(self.world.scripts) 
            }

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if self.continuous_actions:
                space_dim = self.world.dim_n
            else:
                assert self.continuous_actions is False, \
                    'Continuous_actions must be True or False.'
                space_dim = self.world.dim_l

            obs_dim = len(self.scenario.observation(agent, self))
            # 为了和state区分，改为total_observation
            state_dim = len(self.scenario.total_observation(self))
            # action_spaces
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=-1, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            # observation_spaces
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )
        # state_space
        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )
        
        self.steps = 0
        self.win_num = 0
        self.lose_num = 0
        self.stop_steps = {name: self.max_cycles for name in self.agents}

        self.current_actions = [None] * self.num_agents  # agents当前的动作

        self.agents_array = None
        self.scripts_array = None
        self.ag_actions_array = None
        self.sc_actions_array = None
        self.rewards_array = None

    def observation_space(self, agent: AgentID):
        """
        返回单个agent的observation_space
        - ! 这个方法必须对AECenv进行覆写
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID):
        """
        返回单个agent的action_space
        - ! 这个方法必须对AECenv进行覆写
        """
        return self.action_spaces[agent]

    @property
    def num_scripts(self) -> int:
        return len(self.scripts)

    def _seed(self, seed=None):
        """
        随机数种子设置
        """
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent: AgentID):
        """
        返回单个agent的action_space
        - ! 这个方法必须对AECenv进行覆写
        """
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self
        ).astype(np.float32)

    def state(self):
        """
        返回环境的state，为agent.observe的集合
        - ! 这个方法和state_space的设置相一致  
        """
        return self.scenario.total_observation(self).astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        NOTE:
        1. 通过scenario.reset_world进行位置、速度、颜色等的重置
        2. 清空信息存储序列
        3. world.agents, world.scripts的reset由外部scenario进行
        """
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = list.copy(self.possible_agents)
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.steps = 0
        self.win_num = 0
        self.lose_num = 0
        self.stop_steps = {name: self.max_cycles for name in self.agents}

        self.current_actions = [None] * self.num_agents

        if self.scr_on is True:
            self.scripts = list.copy(self.possible_scripts)

        # 添加目标分配算法，应放在reset中
        self.scripts_to_agents = self.target_assignment(self.tar_mode)
        self.agents_to_scripts = dict()
        for key in self.scripts_to_agents:
            item = self.scripts_to_agents[key]
            self.agents_to_scripts[item] = key 

        # array
        count = self.max_cycles + 1
        num_agents = self.num_agents
        num_scripts = self.num_scripts
        self.agents_array = np.zeros((count, num_agents, 6), dtype=float)
        self.scripts_array = np.zeros((count, num_scripts, 6), dtype=float)
        self.ag_actions_array = np.zeros((count, num_agents, 3), dtype=float) \
                                if self.continuous_actions is True else \
                                np.zeros((count, num_agents, 1), dtype=int)
        self.sc_actions_array = np.zeros((count, num_agents, 3), dtype=float) \
                                if self.scr_mode in ["ppo", "ippo", "isac", "iddpg"] else \
                                np.zeros((count, num_agents, 1), dtype=int)
        self.rewards_array = np.zeros((count, num_agents), dtype=float)

        # store initial state
        self._store_state(init_flag=True)

    def _execute_world_step(self):
        """
        Set action for each agent.
        
        NOTE:
        1. current_actions: step()传入的action，current_action[i]
        代表单个agent的动作
        2. action: 单个agent的动作，为array([ ])
        3. action[0:mdim]: 单个agent与move相关的动作
        4. action[midm:]: 单个agent与move无关(communication)的动作
        5. scenario_action: 单个agent实际要执行的动作，为[array([ ])]
        6. sc_actions[i]: 类似于之前的action，为array([ ])  
            sc_actions和current_actions对标
        7. internal_action: 单个script实际要执行的动作，为[array([ ])]

        步骤:
        1. 处理传入的action
                |
        2. world.step()更新每个agent的动作
                |
        3. scenario.reward()获取每个agent的奖励
        - ! 在simple_adversary中local_ratio = None(默认)
            不存在scenario.global_reward()
        """
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]  # 单个agent的动作
            scenario_action = list()  # scenario动作
            scenario_action.append(action)
            self._set_action(scenario_action, agent, self.world)

        for a in self.agents:
            i = self._index_map[a]
            agent = self.world.agents[i]
            # 直接self.ag_actions_array[self.steps] = self.current_actions, 会存储失败
            if self.continuous_actions:
                self.ag_actions_array[self.steps, i] = self.current_actions[i]
            else:
                self.ag_actions_array[self.steps, i] = self.current_actions[i] + 1

        if self.scr_on is True:
            sc_actions = self.scripts_policy()
            for i, script in enumerate(self.world.scripts):
                internal_action = list()
                internal_action.append(sc_actions[i])
                self._set_script_action(internal_action, script)
                
            for s in self.scripts:
                i = self._scripts_map[s]
                script = self.world.scripts[i]
                # 同ag_actions_array
                if self.scr_mode in ["ppo", "ippo", "isac", "iddpg"]:
                    self.sc_actions_array[self.steps, i] = sc_actions[i]
                else:
                    self.sc_actions_array[self.steps, i] = sc_actions[i] + 1

        self.world.step()  # 更新agents.state
        self._store_state()  # 存储agents.state

    def _store_state(self, init_flag=False):
        """
        从_execute_world_step分离出来，以解决无法记录最后一个state的问题
        NOTE:
        1. 不记录动作
        2. 在reset()中调用第一次，存储初始信息
        3. state_array比actions_array多存储一次
        """
        # store agents array
        for a in self.agents:
            i = self._index_map[a]
            agent = self.world.agents[i]
            if init_flag is True:
                self.agents_array[0, i] = agent.unpack
            else:
                self.agents_array[self.steps+1, i] = agent.unpack
        if self.scr_on is True:
            # store scripts array
            for s in self.scripts:
                i = self._scripts_map[s]
                script = self.world.scripts[i]
                if init_flag is True:
                    self.scripts_array[0, i] = script.unpack
                else:
                    self.scripts_array[self.steps+1, i] = script.unpack

    def _set_action(self, action, agent: Agent, world: World):
        """
        Set env action for a particular agent.

        NOTE:
        1. 将action分为c、u进行处理
        2. action <- scenario_action: 
            [array([no_action, move_left, move_right, move_down, move_up])]
        3. agent.accel: 加速，但与加速度有区别，可以理解为权重
        4. action_space: [-1, 1]
        5. 处理逻辑和pettingzoo.mpe相同，如果action=None，则agent.action=0
        """
        if self.continuous_actions:
            if agent.action.u is not None and \
                    isinstance(agent.action.u, (list, np.ndarray)): 
                gamma = agent.action.u[2] 
            else:
                gamma = 0

        agent.action.u = np.zeros(3)

        dt = world.dt
        D2R = np.pi/180  # deg2rad
        R2D = 180/np.pi  # rad2deg
        NX_SCALE = world.constant.NX_SCALE
        NY_SCALE = world.constant.NY_SCALE
        GA_SCALE = world.constant.GA_SCALE

        if action is None:
            return
        if isinstance(action, (np.ndarray, list)):
            if action[0] is None:
                return
            
        if self.continuous_actions:
            # 上一个动作的gamma需要在zeros之前获取
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
            if action[0] == 0:
                agent.action.u[0] = 0
                agent.action.u[1] = 5
                agent.action.u[2] = -np.pi/3
            if action[0] == 1:
                agent.action.u[0] = 2
                agent.action.u[1] = 1
                agent.action.u[2] = 0
            if action[0] == 2:
                agent.action.u[0] = 0
                agent.action.u[1] = 1
                agent.action.u[2] = 0
            if action[0] == 3:
                agent.action.u[0] = -2
                agent.action.u[1] = 1
                agent.action.u[2] = 0
            if action[0] == 4:
                agent.action.u[0] = 0
                agent.action.u[1] = 5
                agent.action.u[2] = np.pi/3
            if action[0] == 5:
                agent.action.u[0] = 0
                agent.action.u[1] = 5
                agent.action.u[2] = 0
            if action[0] == 6:
                agent.action.u[0] = 0
                agent.action.u[1] = -5
                agent.action.u[2] = 0
        # 这个还有使用完action的判断功能
        action = action[1:]  # 如果有action.c，那就是action.c
        # make sure we used all elements of action
        assert len(action) == 0

    def _set_script_action(self, action, script: Script):
        """
        Set env action for a particular script.
        Only discrete action.

        NOTE:
        1. 将action分为c、u进行处理
        2. action <- scenario_action: 
            [array([no_action, move_left, move_right, move_down, move_up])]
        3. agent.accel: 加速，但与加速度有区别，可以理解为权重
        """
        # discrete input: [0]
        # continuous input(from sb3 model): [array([0, 0, 0])] 
        continuous_flag = False
        if action is None:
            return
        if isinstance(action, (np.ndarray, list)):
            if action[0] is None:
                return
            if isinstance(action[0], np.ndarray):
                # action = action[0]  # 连续动s作
                continuous_flag = True
                if script.action.u is not None:
                    gamma = script.action.u[2]
                else:
                    gamma = 0

        script.action.u = np.zeros(3)
        
        world = self.world
        dt = world.dt
        D2R = np.pi/180  # deg2rad
        R2D = 180/np.pi  # rad2deg
        NX_SCALE = world.constant.NX_SCALE
        NY_SCALE = world.constant.NY_SCALE
        GA_SCALE = world.constant.GA_SCALE

        # process discrete action
        # 'nx_s' = [0, 2, 0, -2, 0, 0, 0]
        # 'ny_s' = [5, 1, 1, 1, 5, 5, -5]
        # 'gamma_s' = [-np.pi/3, 0, 0, 0, np.pi/3, 0, 0]
        if continuous_flag:
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
            script.action.u[0] = nx
            script.action.u[1] = ny
            script.action.u[2] = gamma
        else:
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

    def step(self, action):
        """
        step()与外部policy进行交互
        - ! 针对具体环境，应该对该方法进行覆写
        - action不是直接输入给world.agents，而是输入给current_idx;
            current_idx由_agent_selection控制

        NOTE:
        1. cur_agent: agent_name
        2. current_idx: agent_id
        3. next_idx: 下一个agent_id，这个操作使next_idx执行正确
        4. truncations: self.steps >= self.max_cycles时全置true，结束训练
        5. terminations: 没有进行设置
            这个env似乎无法进行评估使用?
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            idx = self._index_map[self.agent_selection]
            self.current_actions[idx] = None 
            self._was_dead_step(action)
            return
        # 分别有agent.idx和agent.name两套循环
        cur_agent = self.agent_selection  # 当前的agent.name
        current_idx = self._index_map[self.agent_selection]  # 当前的agent.idx
        next_idx = (current_idx + 1) % self.num_agents  # 下一个agent.idx

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1

            for a in self.agents:
                self.rewards[a] = self._get_reward(a)
            
            # 已经dead的agent将无法收到global_reward
            global_reward, win = self._get_global_reward()
            for key in self.rewards.keys():
                self.rewards[key] += global_reward
            if self.shared_reward:
                reward = sum(self.rewards.values()) / len(self.rewards)
                self.rewards = {key: reward for key in self.rewards.keys()}

            # store rewards
            for key in self.rewards.keys():
                idx = self._index_map[key]
                self.rewards_array[self.steps-1, idx] = self.rewards[key]

            if win != 2:
                for key in self.infos.keys():
                    self.infos[key]['w'] = win 
            
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
            else:    
                for a in self.agents:
                    self.terminations[a] = self._get_done(a)
            
            iter_agents = list.copy(self.agents)  # self.agents[:]类似于list.copy
            for agent in self.terminations:
                if self.terminations[agent] or self.truncations[agent]:
                    iter_agents.remove(agent)
            self._agent_selector.reinit(iter_agents)
        else:
            self._clear_rewards()

        if self._agent_selector.agent_order:
            self.agent_selection = self._agent_selector.next()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()
        self._deads_step_first()

    def target_assignment(self, tar_mode='auction') -> dict[ScriptID, AgentID]:
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
        scripts_to_agents = dict()

        if tar_mode == 'distance':
            # 根据scipy中的线性规划算法进行距离分配，必须n v n
            red_points = list()
            blue_points = list()
            for a in self.agents:
                agent_idx = self._index_map[a]
                agent = self.world.agents[agent_idx]
                red_points.append(agent.unpack[0:3])
            for s in self.scripts:
                script_idx = self._scripts_map[s]
                script = self.world.scripts[script_idx]
                blue_points.append(script.unpack[0:3])
            red_points = np.array(red_points)
            blue_points = np.array(blue_points)
            distances = np.linalg.norm(red_points[:, np.newaxis] - blue_points, axis=2)
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(distances)
            for i in range(len(self.scripts)):    
                scripts_to_agents[self.scripts[row_ind[i]]] = self.agents[col_ind[i]] 
        elif tar_mode == 'auction':
            # 拍卖算法
            value_function = lambda s, a: \
                             self.scenario.single_close_reward(self.world.scripts[self._scripts_map[s]].unpack,
                                                               self.world.agents[self._index_map[a]].unpack)
            scripts_to_agents = Auction(value_function, self.possible_scripts, self.possible_agents).run()
        return scripts_to_agents

    def scripts_policy(self):
        """
        The policy for all scripts.
        This method should not be called when 
        self.scr_on is not True.

        action mode:
        1. single mini_max
        2. single max_max

        NOTE:
        1. 当scripts_to_agents对应的agent被清理时，self.scripts相应被清理
        2. 被清理的script(物理意义上没有)，对应的sc_actions为None
        3. scripts_to_agents不会被清理
        """
        # scripts必须存在，否则该方法不应该被调用
        assert self.scr_on is True, \
            'This method should not be called.'

        for s in self.scripts:
            a = self.scripts_to_agents[s]
            if a not in self.agents:
                self.scripts.remove(s)

        # 采取len(self._scripts_map)，因为self._scripts_map不会变化
        sc_actions: list[Any]  # 为了解决警告
        sc_actions = [None] * len(self._scripts_map)

        assert self.scr_mode is not None, 'Must get scr_mode.'

        if self.scr_mode == 'steady':
            # 即使是固定动作，也根据self.scripts赋值
            for s in self.scripts:
                script_idx = self._scripts_map[s]
                sc_actions[script_idx] = 2
        elif self.scr_mode == 'random':
            for s in self.scripts:
                script_idx = self._scripts_map[s]
                sc_actions[script_idx] = self.np_random.integers(0, 6)
        elif self.scr_mode == 'mini_max':
            for s in self.scripts:
                a = self.scripts_to_agents[s]
                agent_idx = self._index_map[a]
                script_idx = self._scripts_map[s]
                agent = self.world.agents[agent_idx]
                script = self.world.scripts[script_idx]
                # 以blue为主体
                simu_r = agent.unpack
                simu_b = script.unpack
                rewards_r = []
                rewards_b = []
                # red: agent, blue: script
                for i in range(0, 7):
                    action_r = i
                    simu_r_next = self.world.imitate_action(action_r, simu_r)
                    rewards_r.append(self.scenario.single_close_reward(simu_b, simu_r_next, self.steps))
                action_r = rewards_r.index(min(rewards_r))  # 这里的rewards_r实际是blue的奖励
                simu_r_next = self.world.imitate_action(action_r, simu_r)
                for i in range(0, 7):
                    action_b = i
                    simu_b_next = self.world.imitate_action(action_b, simu_b)
                    rewards_b.append(self.scenario.single_close_reward(simu_b_next, simu_r_next, self.steps))
                action_b = rewards_b.index(max(rewards_b))
                sc_actions[script_idx] = action_b
        elif self.scr_mode == 'max_max':
            for s in self.scripts:
                a = self.scripts_to_agents[s]
                agent_idx = self._index_map[a]
                script_idx = self._scripts_map[s]
                agent = self.world.agents[agent_idx]
                script = self.world.scripts[script_idx]
                # 以blue为主体
                simu_r = agent.unpack
                simu_b = script.unpack
                rewards_r = []
                rewards_b = []
                # red: agent, blue: script
                for i in range(0, 7):
                    action_r = i
                    simu_r_next = self.world.imitate_action(action_r, simu_r)
                    rewards_r.append(self.scenario.single_close_reward(simu_r_next, simu_b, self.steps))
                action_r = rewards_r.index(max(rewards_r))
                simu_r_next = self.world.imitate_action(action_r, simu_r)
                for i in range(0, 7):
                    action_b = i
                    simu_b_next = self.world.imitate_action(action_b, simu_b)
                    rewards_b.append(self.scenario.single_close_reward(simu_b_next, simu_r_next, self.steps))
                action_b = rewards_b.index(max(rewards_b))
                sc_actions[script_idx] = action_b
        elif self.scr_mode in ["ppo", "ippo", "isac", "iddpg"]:
            if hasattr(self.scenario, "b_model_policy") is False:
                raise Exception("Must have b_model_policy.")
            for s in self.scripts:
                a = self.scripts_to_agents[s]
                agent_idx = self._index_map[a]
                script_idx = self._scripts_map[s]
                agent = self.world.agents[agent_idx]
                script = self.world.scripts[script_idx]
                sc_actions[script_idx] = self.scenario.b_model_policy(script, agent, self)
        return sc_actions
    
    def monitor(self):  
        """
        The monitor of env.
        """
        pass
