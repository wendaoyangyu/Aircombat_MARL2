from typing import Optional, Union
import numpy as np


D2R = np.pi / 180  # deg2rad
R2D = 180 / np.pi  # rad2deg


class CloseConstant:
    """
    The constant for WVR.
    """
    A_MAX = 180
    D_MAX = 20000
    DH_MAX = 10000
    DV_MAX = 500
    Z_MAX = 10000
    Z_MIN = 1000
    V_MAX = 500
    V_MIN = 50

    NX_SCALE = 2
    NY_SCALE = 5
    GA_SCALE = np.pi  # math.pi/2.25


class FarConstant:
    """
    The constant for BVR.
    """
    A_MAX = 180
    D_MAX = 200000
    DH_MAX = 100000
    DV_MAX = 500
    Z_MAX = 20000
    Z_MIN = 1000
    V_MAX = 500
    V_MIN = 50
    DD_MAX = 1000

    NX_SCALE = 2
    NY_SCALE = 5
    GA_SCALE = np.pi  # math.pi/2.25

    MD_TYPE = 2


def angle_limit(angle):
    if angle > 180:
        angle = angle - 2 * 180
    elif angle < -180:
        angle = angle + 2 * 180
    return angle


def safe_load(data: dict, attr: str, init: Union[int, list]):
    if data.get(attr) is not None:
        return data.get(attr)
    else:
        return init


class EntityState:  
    """
    Physical/external base state of all aircraft.
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.phi = None
        self.mu = None
        self.v = None


class AgentState(EntityState):  
    """
    State of agents (including communication and internal/mental state).
    """
    def __init__(self):
        super().__init__()


class Action:  
    """
    Action of the aircraft.
    """
    def __init__(self):
        # physical action
        self.u = None


class Entity:
    """
    This class is the base class for agents, 
    containing their common attributes.
    - ! del landmarks now.

    NOTE: 
    1. unpack在Agent、Script中都会生效
    2. 添加is_alive属性，且开始时为True
    (和mpe一致，e.g. self.movable = False)
    """
    def __init__(self):
        # name
        self.name = '' 
        # state
        self.state = EntityState()
        # action
        self.action = Action()
        # alive
        self.is_alive = True

    def initial_state(self, initial_simu: np.ndarray):
        # only can be used one time
        x, y, z, phi, mu, v = initial_simu
        self.state.x = x
        self.state.y = y
        self.state.z = z
        self.state.phi = phi
        self.state.mu = mu
        self.state.v = v

    @ property
    def unpack(self): 
        """
        Return simu of entity.
        simu: [x, y, z, phi, mu, v]

        NOTE:
        1. 不要加括号
        2. 修改返回的simu中的值，不会对entity本身造成影响
        """
        simu = np.array([self.state.x, self.state.y, self.state.z, 
                         self.state.phi, self.state.mu, self.state.v])
        return simu


class Agent(Entity):  
    """
    Properties of agent entities.
    """
    def __init__(self):
        super().__init__()
        # script behavior to execute
        self.action_callback = None 
        # state
        self.state = AgentState()


class Script(Entity):  
    """
    Only script, not agent.
    """
    def __init__(self):
        super().__init__()
        # state
        self.state = AgentState()


class World:  
    """
    Multi-agent world.

    NOTE:
    1. step()针对单次动作(这次动作包含所有存在动作的entity)运行
    2. 当前step()的执行逻辑是先将agents全部运行，再运行scripts
    3. 所有输入动作的缩放和裁剪都在base_env中完成
    """

    agents: list[Agent]
    scripts: list[Script]

    def __init__(self):
        """
        List of agents and entities (can change at execution-time!)
        """
        self.agents = []  # 具体放置core.Agent()
        self.scripts = []  # core.Script()
        # position dimensionality
        # TODO: 这个维度应该指的是[nx, ny, gamma]
        self.dim_n = 3  # 连续动作空间维度 [nx, ny, gamma]
        self.dim_l = 7  # 离散动作空间维度 机动动作库(library)
        # gravity
        self.gravity = 9.8
        # pi
        self.pi = np.pi
        # wvr
        self.close = True
        # constant
        if self.close is True:
            self.constant = CloseConstant()
            # simulation timestep
            self.dt = 0.5
        else:
            assert self.close is False, \
                'Close must be True or False.'
            self.constant = FarConstant()
            # simulation timestep
            self.dt = 2

    @property
    def entities(self):
        """
        Return all entities in the world.
        """
        return self.agents + self.scripts
    
    @property
    def policy_agents(self):
        """
        Return all agents controllable by external policies.
        """
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        """
        Return all agents controlled by world scripts.
        """
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        """
        Update state of the world.

        NOTE:
        1. scripted_agents: 由world内部脚本控制的agents
        2. policy_agents: 由外部policy控制的agents
        3. scripts: 不能由外部控制的纯脚本
        4. agent.action.u: nx, ny, gamma
        5. a_force(agent): [[nx, ny, gamma], [nx, ny, gamma]...]
        6. s_force(script): [[nx, ny, gamma], [nx, ny, gamma]...]

        步骤:
        1. apply_action_force: 给每个agent的action
                        |
        2. X apply_environment_force: 对force进行碰撞处理(get_collision_force)
                        |
        3. integrate_state: 应用物理(p)动作
                        |            
        4. X update_agent_state: 应用交流(c)动作
                        |  如果存在scripts
        5. script_action_force: 给每个script的action
                        |
        6. integrate_state: 应用物理(p)动作
        - ! 如果is_alive=False，将不执行动作
        """
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to agents
        a_force = [None] * len(self.agents)
        # apply agent physical controls
        a_force = self.apply_action_force(a_force)
        # integrate physical state
        self.integrate_state(a_force)

        # scripts control
        # 这里必须和agents拆开运行(为了安全)
        # 删去if len(self.scripts) != 0 应该也可以正确运行
        # 这种方式或许能提高运行效率
        if len(self.scripts) != 0:
            # gather forces applied to scripts
            s_force = [None] * len(self.scripts)
            # apply agent physical controls
            s_force = self.script_action_force(s_force)
            # integrate physical state
            self.script_integrate(s_force)

    def apply_action_force(
        self,
        a_force: list[Optional[list]]
    ) -> list[list]:
        """
        Only agent.action.u -> a_force.
        """
        # set applied forces
        for i, agent in enumerate(self.agents):
            a_force[i] = agent.action.u
        return a_force

    def integrate_state(
        self,
        a_force: list[list]
    ) -> None:
        """
        Integrate physical state.
        """
        g = self.gravity
        pi = self.pi
        tau = self.dt
        sin = np.sin
        cos = np.cos
        V_MIN = self.constant.V_MIN
        V_MAX = self.constant.V_MAX
        Z_MIN = self.constant.Z_MIN
        Z_MAX = self.constant.Z_MAX
        for i, agent in enumerate(self.agents):
            if agent.is_alive is True:
                x, y, z, phi, mu, v = agent.unpack
                nx, ny, gamma = a_force[i]
                # dynamics
                v_dot = g * (nx - sin(mu * D2R))
                mu_dot = g * (ny * cos(gamma) - cos(mu * D2R)) / v
                phi_dot = - g * ny * sin(gamma) / v / cos(mu * D2R)
                # dynamic correction
                mu_dot = np.clip(mu_dot, -pi/6, pi/6)
                phi_dot = np.clip(phi_dot, -pi/6, pi/6)
                # dynamic update
                v = v + v_dot * tau
                mu = mu + mu_dot * tau * R2D
                phi = phi + phi_dot * tau * R2D
                # dynamic correct
                v = np.clip(v, V_MIN, V_MAX)
                mu = angle_limit(mu)
                phi = angle_limit(phi)
                # kinematical
                x_dot = v * cos(mu * D2R) * cos(phi * D2R)
                y_dot = v * cos(mu * D2R) * sin(phi * D2R)
                z_dot = v * sin(mu * D2R)
                # kinematical update
                x = x + x_dot * tau
                y = y + y_dot * tau
                z = z + z_dot * tau
                # kinematical correct
                z = np.clip(z, Z_MIN, Z_MAX)
                # storage
                agent.state.x = x
                agent.state.y = y
                agent.state.z = z
                agent.state.mu = mu  # mu:航倾角(航迹倾斜角)
                agent.state.phi = phi  # phi:航向角(航迹方位角)
                agent.state.v = v  # v:速度
            else:
                assert agent.is_alive is False, 'Agent.is_alive must be bool.'
                continue

    def script_action_force(
        self,
        s_force: list[Optional[list]]
    ) -> list[list]:
        """
        Only script.action.u -> p_force.
        """
        # set applied forces
        for i, script in enumerate(self.scripts):
            s_force[i] = script.action.u
        return s_force

    def script_integrate(
        self,
        s_force: list[list]
    ) -> None:
        """
        Integrate physical state.
        """
        g = self.gravity
        pi = self.pi
        tau = self.dt
        sin = np.sin
        cos = np.cos
        V_MIN = self.constant.V_MIN
        V_MAX = self.constant.V_MAX
        Z_MIN = self.constant.Z_MIN
        Z_MAX = self.constant.Z_MAX
        for i, script in enumerate(self.scripts):
            if script.is_alive is True:
                x, y, z, phi, mu, v = script.unpack
                nx, ny, gamma = s_force[i]
                # dynamics
                v_dot = g * (nx - sin(mu * D2R))
                mu_dot = g * (ny * cos(gamma) - cos(mu * D2R)) / v
                phi_dot = - g * ny * sin(gamma) / v / cos(mu * D2R)
                # dynamic correction
                mu_dot = np.clip(mu_dot, -pi/6, pi/6)
                phi_dot = np.clip(phi_dot, -pi/6, pi/6)
                # dynamic update
                v = v + v_dot * tau
                mu = mu + mu_dot * tau * R2D
                phi = phi + phi_dot * tau * R2D
                # dynamic correct
                v = np.clip(v, V_MIN, V_MAX)
                mu = angle_limit(mu)
                phi = angle_limit(phi)
                # kinematical
                x_dot = v * cos(mu * D2R) * cos(phi * D2R)
                y_dot = v * cos(mu * D2R) * sin(phi * D2R)
                z_dot = v * sin(mu * D2R)
                # kinematical update
                x = x + x_dot * tau
                y = y + y_dot * tau
                z = z + z_dot * tau
                # kinematical correct
                z = np.clip(z, Z_MIN, Z_MAX)
                # storage
                script.state.x = x
                script.state.y = y
                script.state.z = z
                script.state.mu = mu  # mu:航倾角(航迹倾斜角)
                script.state.phi = phi  # phi:航向角(航迹方位角)
                script.state.v = v  # v:速度
            else:
                assert script.is_alive is False, 'Script.is_alive must be bool.'
                continue

    def imitate_action(self, index, simu):
        """
        For mini_max: simu -> index -> simu_.
        只是模拟动作，不会对entity产生影响
        
        NOTE:
        1. simu: np.array([x, y, z, phi, mu, v])
        2. action: [nx, ny, gamma]
        3. simu必须独立于entity.state
        4. 必须输入world类型，因为会影响参数
        """
        g = self.gravity
        pi = self.pi
        tau = self.dt
        sin = np.sin
        cos = np.cos
        V_MIN = self.constant.V_MIN
        V_MAX = self.constant.V_MAX
        Z_MIN = self.constant.Z_MIN
        Z_MAX = self.constant.Z_MAX

        # set action
        action = []
        if index == 0:
            action = [0, 5, -np.pi/3]
        if index == 1:
            action = [2, 1, 0]
        if index == 2:
            action = [0, 1, 0]
        if index == 3:
            action = [-2, 1, 0]
        if index == 4:
            action = [0, 5, np.pi/3]
        if index == 5:
            action = [0, 5, 0]
        if index == 6:
            # 以前的env，这里根据phi的不同，会有不同的参数
            # 但是这里gamma = 0，没有正负之分
            action = [0, -5, 0]
        
        # apply action
        x, y, z, phi, mu, v = np.copy(simu)
        nx, ny, gamma = list.copy(action)
        # dynamics
        v_dot = g * (nx - sin(mu * D2R))
        mu_dot = g * (ny * cos(gamma) - cos(mu * D2R)) / v
        phi_dot = - g * ny * sin(gamma) / v / cos(mu * D2R)
        # dynamic correction
        mu_dot = np.clip(mu_dot, -pi/6, pi/6)
        phi_dot = np.clip(phi_dot, -pi/6, pi/6)
        # dynamic update
        v = v + v_dot * tau
        mu = mu + mu_dot * tau * R2D
        phi = phi + phi_dot * tau * R2D
        # dynamic correct
        v = np.clip(v, V_MIN, V_MAX)
        mu = angle_limit(mu)
        phi = angle_limit(phi)
        # kinematical
        x_dot = v * cos(mu * D2R) * cos(phi * D2R)
        y_dot = v * cos(mu * D2R) * sin(phi * D2R)
        z_dot = v * sin(mu * D2R)
        # kinematical update
        x = x + x_dot * tau
        y = y + y_dot * tau
        z = z + z_dot * tau
        # kinematical correct
        z = np.clip(z, Z_MIN, Z_MAX)
        # simu_
        simu_ = np.array([x, y, z, phi, mu, v])
        return simu_
    
if __name__ == '__main__':
    a1 = Agent()
    a1.state.mu = 1
    mu1 = a1.unpack[4]
    mu1 = 22
    print(a1.unpack)
    print(mu1)
