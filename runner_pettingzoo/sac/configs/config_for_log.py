def operate_config(args, run_dir: str):
    # run_dir[-1]
    last_char = run_dir[-1]
    last_int = int(last_char)

    # _前的数字
    rev_dir = run_dir[::-1]
    underscore_find = rev_dir.rfind('_')
    if underscore_find != -1:
        last_char = rev_dir[underscore_find + 1]  # '_'前的数字
        last_int = int(last_char)

    if args.scenario_name == 'simple_2v2':
        if last_int == 1:
            args.scr_mode = 'steady'
            args.tar_mode = 'distance'
            args.continuous_actions = False
        elif last_int == 2:
            args.scr_mode = 'random'
            args.tar_mode = 'distance'
            args.continuous_actions = False
        elif last_int == 3:
            args.scr_mode = 'mini_max'
            args.tar_mode = 'auction'
            args.continuous_actions = False
        elif last_int == 4:
            args.scr_mode = 'steady'
            args.tar_mode = 'distance'
            args.continuous_actions = True
        elif last_int == 5:
            args.scr_mode = 'random'
            args.tar_mode = 'distance'
            args.continuous_actions = True
        elif last_int == 6:
            args.scr_mode = 'mini_max'
            args.tar_mode = 'auction'
            args.continuous_actions = True
            args.simu_init = [[0, 0, 3000, 45, 0, 200],
                              [1000, 1000, 3000, 45, 0, 200],
                              [3000, 3000, 3000, 90, 0, 220],
                              [4000, 4000, 3000, 90, 0, 220]]
        elif last_int == 8:
            args.scr_mode = 'ppo' # sac
            args.tar_mode = 'auction'
            args.continuous_actions = True
            args.simu_init = [[0, 0, 3000, 180, 0, 200],
                              [1000, 1000, 3000, 180, 0, 200],
                              [1000, 5000, 3000, 0, 0, 220],
                              [2000, 6000, 3000, 0, 0, 220]]
    else:
        raise NotImplementedError
    return args
