def make_env(all_args):
    if all_args.scenario_name == 'simple_2v2':
        from envs.mae_parallel.simple_2v2 import raw_env  # mae从外部envs中导入
        env = raw_env(all_args.num_agents, continuous_actions=all_args.continuous_actions,
                        scr_mode=all_args.scr_mode)
        return env
    if all_args.scenario_name == 'simple_2v2_point':
        from envs.mae_parallel.simple_2v2_point import raw_env  # mae从外部envs中导入
        env = raw_env(all_args.num_agents, continuous_actions=all_args.continuous_actions,
                        scr_mode=all_args.scr_mode)
        return env
    