from typing import Optional
import numpy as np
from envs.mae_parallel.mae_utils.core import Entity, World


D2R = np.pi / 180  # deg2rad
R2D = 180 / np.pi  # rad2deg


class BaseScenario:  
    """
    Defines scenario upon which the world is built.
    
    DO:
    1. add __init__()
    2. add single_situation()
    3. 将single系列的方法的输入参数都均改为simu

    Add(mae):
    1. benchmark_data
    2. done
    3. del total_obs     
    """

    def __init__(self):
        """
        Must be overwritten.
        """
        self.N = None
        self.max_cycles = None

    def make_world(self):  
        """
        Create elements of the world.
        """
        raise NotImplementedError()

    def reset_world(self, world, np_random):  
        """
        Create initial conditions of the world.
        """
        raise NotImplementedError()

    def benchmark_data(self, agent, env):
        """
        Return the info of single agent.
        """
        return {}

    def observation(self, agent, env):
        """
        Return the observation of single agent.
        """
        return np.zeros(0)

    def done(self, agent, env):
        """
        Return the observation of single agent.
        """
        return False

    def reward(self, agent, env):
        """
        Return the reward from single agent.
        """
        return 0

    def global_reward(self, env):
        """
        Return the reward from global env.
        """
        return 0, 2

    @staticmethod
    def single_situation(simu_r: np.ndarray, simu_b: np.ndarray):
        """
        Compute situation for single red v blue.
        red, blue can exchange.
        distance, delta_h, delta_v, q_r, q_b, beta_rb.
        """
        sin = np.sin
        cos = np.cos
        acos = np.arccos
        sqrt = np.sqrt
        # red
        x_r, y_r, z_r, phi_r, mu_r, v_r = simu_r
        x_b, y_b, z_b, phi_b, mu_b, v_b = simu_b
        # relative distance
        distance = sqrt((x_r - x_b) ** 2 + (y_r - y_b) ** 2 + (z_r - z_b) ** 2)
        assert distance != 0, print("Distance can't be zero!")
        # ATA[0, pi]
        q_r_temp = ((x_b - x_r) * cos(phi_r * D2R) * cos(mu_r * D2R) +
                    (y_b - y_r) * sin(phi_r * D2R) * cos(mu_r * D2R) +
                    (z_b - z_r) * sin(mu_r * D2R)) / distance
        q_r_temp = np.clip(q_r_temp, -1, 1)
        q_r = acos(q_r_temp) * R2D
        # AA[0, pi]
        q_b_temp = ((x_b - x_r) * cos(phi_b * D2R) * cos(mu_b * D2R) +
                    (y_b - y_r) * sin(phi_b * D2R) * cos(mu_b * D2R) +
                    (z_b - z_r) * sin(mu_b * D2R)) / distance
        q_b_temp = np.clip(q_b_temp, -1, 1)
        q_b = acos(q_b_temp) * R2D
        # the included angle of the velocity vector
        beta_rb_temp = (cos(mu_r * D2R) * cos(phi_r * D2R) * cos(mu_b * D2R) * cos(phi_b * D2R) +
                        cos(mu_r * D2R) * sin(phi_r * D2R) * cos(mu_b * D2R) * sin(phi_b * D2R) +
                        sin(mu_r * D2R) * sin(mu_b * D2R))
        beta_rb_temp = np.clip(beta_rb_temp, -1, 1)
        beta_rb = acos(beta_rb_temp) * R2D
        # delta_h
        delta_h = z_r - z_b
        delta_v = v_r - v_b
        return np.array([distance, delta_h, delta_v, q_r, q_b, beta_rb], dtype=np.float32)
    
    def single_close_reward(
        self, 
        simu_r: np.ndarray, 
        simu_b: np.ndarray, 
        step_num: Optional[int] = None,
        end_on=True
    ) -> float:
        """
        Compute single_close_reward for single red v blue.
        red, blue can exchange.  
        
        Note:
        1. Need max_cycles，该方法必须scenario被覆写后才可以调用
        2. 如果以Entity作为输出参数的话，该方法将无法被mini_max调用
        3. 因为MA的复杂性，和ad_env中reward获取不同，这里不返回win
        """
        assert self.max_cycles is not None, 'Need int -> max_cycles.'
        exp = np.exp
        # red
        x_r, y_r, z_r, phi_r, mu_r, v_r = simu_r
        x_b, y_b, z_b, phi_b, mu_b, v_b = simu_b 
        # 获取态势
        obs = self.single_situation(simu_r, simu_b)  # distance, delta_h, delta_v, q_r, q_b
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
        if end_on:
            if (100 <= distance <= 1000) and (q_r <= 30) and (q_b <= 60):  # q_b < 60
                r_end = 20
            elif (100 <= distance <= 1000) and (q_r >= 120) and (q_b >= 150):  # q_b > 150
                r_end = -20
            elif step_num == self.max_cycles:
                r_end = -20
            else:
                r_end = 0
        else:
            assert end_on is False, 'The end_flag must be bool.'
            r_end = 0
        # 综合奖励
        w_a = 0.45  # r_a
        w_d = 0.25  # r_d 
        w_h = 0.15  # r_h
        w_v = 0.15  # r_v
        r_inside = w_a * r_a + w_d * r_d + w_h * r_h + w_v * r_v + r_p + r_s
        r_total = r_inside + r_end
        return r_total

    def mini_max(
        self, 
        red: Entity,
        blue: Entity,
        world: World,
        steps: int
    ) -> int:
        # 以red为主体
        simu_r = red.unpack
        simu_b = blue.unpack
        rewards_r = []
        rewards_b = []
        # red: agent, blue: script
        for i in range(0, 7):
            action_b = i
            simu_b_next = world.imitate_action(action_b, simu_b)
            rewards_b.append(self.single_close_reward(simu_b_next, simu_r, steps))
        action_b = rewards_b.index(max(rewards_b))
        simu_b_next = world.imitate_action(action_b, simu_b)
        for i in range(0, 7):
            action_r = i
            simu_r_next = world.imitate_action(action_r, simu_r)
            rewards_r.append(self.single_close_reward(simu_r_next, simu_b_next, steps))
        action_r = rewards_r.index(max(rewards_r))
        return action_r
    