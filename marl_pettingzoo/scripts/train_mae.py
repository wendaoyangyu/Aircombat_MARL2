"""Uses Stable-Baselines3 to train agents in the Knights-Archers-Zombies environment using SuperSuit vector envs.

This environment requires using SuperSuit's Black Death wrapper, to handle agent death.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import numpy as np

import os
import supersuit as ss
# from stable_baselines3 import PPO

from pettingzoo.butterfly import knights_archers_zombies_v10
# from runner_pettingzoo.ppo.configs.base_config import BaseConfig
# from runner_pettingzoo.ppo.configs.mae_config import MAEConfig
# from typing import Union

# def train(all_args: Union[BaseConfig, MAEConfig]): # 为了统一其他算法的通用性
def train(all_args):
    # Train a single model to play as each agent in an AEC environment
    env_kwargs = dict()
    if all_args.env_name == "pettingzoo":
        if all_args.scenario_name == "knights_archers_zombies_v10":
            env_fn = knights_archers_zombies_v10
            env_kwargs = dict(max_cycles=all_args.episode_limit, max_zombies=all_args.max_zombies, 
                              vector_state=all_args.vector_state)
    elif all_args.env_name == "MAE":
        if all_args.scenario_name == "simple_2v2":
            import envs.mae.simple_2v2 as simple_2v2 
            env_fn = simple_2v2
            env_kwargs = dict(max_cycles=all_args.episode_limit, continuous_actions=all_args.continuous_actions,
                              scr_mode=all_args.scr_mode, shared_reward=all_args.shared_reward, tar_mode=all_args.tar_mode,
                              simu_init=all_args.simu_init, use_seed=all_args.use_seed, selfplay=all_args.selfplay)

    env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    # Pre-process using SuperSuit
    if env.unwrapped.metadata.get('name') == "knights_archers_zombies_v10":
        visual_observation = not env.unwrapped.vector_state
    else:
        visual_observation = False

    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=all_args.seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    # Use a CNN policy if the observation space is visual
    model = None
    if all_args.selfplay or all_args.curriculum_learning or all_args.transfer_learning:
        if all_args.algorithm == 'ippo':
            from marl_pettingzoo.ppo.ppo import PPO
            model_path = '../../runner_pettingzoo/ppo/sub_model/simple_2v2_ippo.pkl' 
            model = PPO.load(model_path,env,verbose=3,batch_size=256,gamma=all_args.gamma)
        elif all_args.algorithm == 'isac':
            from marl_pettingzoo.sac.sac import SAC
            model_path = '../../runner_pettingzoo/sac/sub_model/simple_2v2_isac.pkl' 
            model = SAC.load(model_path,env,verbose=3,batch_size=256,gamma=all_args.gamma)
        elif all_args.algorithm == 'iddpg':
            from marl_pettingzoo.ddpg.ddpg import DDPG
            model_path = '../../runner_pettingzoo/ddpg/sub_model/simple_2v2_iddpg.pkl' 
            model = DDPG.load(model_path,env,verbose=3,batch_size=256,gamma=all_args.gamma)
    else:
        if all_args.algorithm == 'ippo':
            from marl_pettingzoo.ppo.ppo import PPO
            from stable_baselines3.ppo import CnnPolicy, MlpPolicy
            model = PPO(
                CnnPolicy if visual_observation else MlpPolicy,
                env,
                verbose=3,
                batch_size=256,
                gamma=all_args.gamma
            )
        elif all_args.algorithm == 'isac':
            from marl_pettingzoo.sac.sac import SAC
            from stable_baselines3.sac import CnnPolicy, MlpPolicy
            model = SAC(
                CnnPolicy if visual_observation else MlpPolicy,
                env,
                verbose=3,
                batch_size=2048,
                gamma=all_args.gamma
            )
        elif all_args.algorithm == 'iddpg':
            from marl_pettingzoo.ddpg.ddpg import DDPG
            from stable_baselines3.ddpg import CnnPolicy, MlpPolicy
            model = DDPG(
                CnnPolicy if visual_observation else MlpPolicy,
                env,
                verbose=3,
                batch_size=256,
                gamma=all_args.gamma
            )
    
    model.learn(total_timesteps=all_args.max_train_steps,progress_bar=True)

    # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
    model.save(all_args.save_path)

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


# def eval(all_args: Union[BaseConfig, MAEConfig]):
def eval(all_args):
    # Train a single model to play as each agent in an AEC environment
    env_kwargs = dict()
    if all_args.env_name == "pettingzoo":
        if all_args.scenario_name == "knights_archers_zombies_v10":
            env_fn = knights_archers_zombies_v10
            env_kwargs = dict(max_cycles=all_args.episode_limit, max_zombies=all_args.max_zombies, 
                              vector_state=all_args.vector_state)
    elif all_args.env_name == "MAE":
        if all_args.scenario_name == "simple_2v2":
            import envs.mae.simple_2v2 as simple_2v2 
            env_fn = simple_2v2
            env_kwargs = dict(max_cycles=all_args.episode_limit, continuous_actions=all_args.continuous_actions,
                              scr_mode=all_args.scr_mode, shared_reward=all_args.shared_reward, tar_mode=all_args.tar_mode,
                              simu_init=all_args.simu_init, use_seed=all_args.use_seed)
        
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=all_args.render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    if env.unwrapped.metadata.get('name') == "knights_archers_zombies_v10":
        visual_observation = not env.unwrapped.vector_state
    else:
        visual_observation = False
        
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={all_args.num_games_eval}, render_mode={all_args.render_mode})"
    )

    try:
        # latest_policy = max(
        #     glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        # )
        latest_policy = all_args.load_path + '.pkl'
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = None
    if all_args.algorithm == 'ippo':
        from marl_pettingzoo.ppo.ppo import PPO
        model = PPO.load(latest_policy)
    elif all_args.algorithm == 'isac':
        from marl_pettingzoo.sac.sac import SAC
        model = SAC.load(latest_policy)
    elif all_args.algorithm == 'iddpg':
        from marl_pettingzoo.ddpg.ddpg import DDPG
        model = DDPG.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    num_games = all_args.num_games_eval

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                # act = env.action_space(agent).sample() 不进行sample
                act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)

    if all_args.env_name == 'MAE':
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                # act = env.action_space(agent).sample() 不进行sample
                act = model.predict(obs, deterministic=True)[0]
            env.step(act)

        fig_path = all_args.load_path
        idx = fig_path.rfind('/')
        if idx != -1:
            fig_path = fig_path[:idx]
            fig_path = fig_path + '/fig'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        env.env.env.monitor(all_args.en, all_args.only_fig, fig_path, all_args.sub_fig)  # TODO: fix this
        env.close()
        data = np.load(all_args.load_path + '.npz')
        if all_args.algorithm == 'ippo':
            from marl_pettingzoo.ppo.utils.sb3_plot import plot
        elif all_args.algorithm == 'isac':
            from marl_pettingzoo.sac.utils.sb3_plot import plot
        elif all_args.algorithm == 'iddpg':
            from marl_pettingzoo.ddpg.utils.sb3_plot import plot
        plot(data, all_args.en, all_args.only_fig, fig_path)
        env.close()
    return avg_reward


if __name__ == "__main__":
    import os, sys  
    os.chdir(sys.path[0])  
    from runner_pettingzoo.ppo.configs.kaz_config import KAZConfig
    args = KAZConfig('knights_archers_zombies_v10', 'ippo')
    args.max_train_steps = int(10000)  # 81_920
    args.max_zombies = 50

    # train(args)

    # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
    args.render_mode = None
    args.num_games_eval = 10
    eval(args)

    # Watch 2 games (takes ~10 seconds on a laptop CPU)
    args.render_mode = "human"
    args.num_games_eval = 2
    eval(args)
    