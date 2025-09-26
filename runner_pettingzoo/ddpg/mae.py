import os, sys  
os.chdir(sys.path[0])  
from marl_pettingzoo.scripts.train_mae import train, eval
from runner_pettingzoo.ddpg.configs.mae_config import MAEConfig

TRAIN_ID = 'n'
ENV_ID = 'simple_2v2'  # simple_2v2
ALGO_ID = 'iddpg'  # rmasac


def get_simu_init(id: int):
    simu_init = None
    if id == 1:
        simu_init = [[0, 0, 3000, 45, 0, 200],
                     [1000, 1000, 3000, 45, 0, 200],
                     [3000, 3000, 3000, 90, 0, 220],
                     [4000, 4000, 3000, 90, 0, 220]]
    elif id == 2:
        simu_init = [[0, 0, 3000, 40, 0, 200],
                     [3000, 0, 3000, 40, 0, 200],
                     [3000, 3000, 3000, -120, 0, 220],
                     [6000, 3000, 3000, -120, 0, 220]]
    elif id == 3: # 红方态势占优
        simu_init = [[0, 0, 4000, 45, 0, 200],
                     [1000, 1000, 4000, 45, 0, 200],
                     [3000, 3000, 3000, 90, 0, 220],
                     [4000, 4000, 3000, 90, 0, 220]]
    elif id == 4: # 红蓝态势均等
        simu_init = [[0, 0, 3000, 180, 0, 200],
                     [1000, 1000, 3000, 180, 0, 200],
                     [1000, 5000, 3000, 0, 0, 220],
                     [2000, 6000, 3000, 0, 0, 220]]
    elif id == 5: # 蓝方态势占优
        simu_init = [[0, 0, 2000, 60, 0, 200],
                     [3000, 0, 2000, 60, 0, 200],
                     [3000, 3000, 3000, -120, 0, 220],
                     [6000, 3000, 3000, -120, 0, 220]]
    if simu_init is None:
        raise Exception("simu_init is None")
    return simu_init


if __name__ == "__main__":
    args = MAEConfig(ENV_ID, ALGO_ID)
    args.max_train_steps = int(5000)  # 200e4
    args.shared_reward = False
    args.tar_mode = 'auction'
    args.scr_mode = 'mini_max' # mini_max / steady / ppo / iddpg
    if args.scr_mode == 'steady':
        args.tar_mode = 'distance'
    args.gamma = 0.99
    args.continuous_actions = True

    """
    TODO:
    1. 若需要进行自博弈训练，则需要将elif args.scr_mode == "isac"补上
    """
    if args.scr_mode in ["ppo", "ippo", "isac", "iddpg"]:
        args.simu_init = get_simu_init(1) # 2/3/4/5
    elif args.scr_mode == "steady":
        args.simu_init = get_simu_init(3)
    elif args.scr_mode == "random":
        args.simu_init = get_simu_init(4)
    elif args.scr_mode == "mini_max":
        args.simu_init = get_simu_init(5)

    args.use_seed = False
    args.seed = 123
    args.en = False
    args.only_fig = True  # True: 保存图片但不显示，False: 只显示图片
    args.sub_fig = True  # True: 额外的绘图
    args.selfplay = False  # False: 普通对战, True: 自我对战, 需要和args.scr_mode同时修改
    args.curriculum_learning = False  # False: 不使用课程学习, True: 使用
    args.transfer_learning = False  # False: 不使用迁移学习, True: 使用

    if TRAIN_ID == 'y':
        train(args)
    elif TRAIN_ID == 'n':
        # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
        args.render_mode = None
        args.num_games_eval = 2  # 10
        eval(args)

        # # Watch 2 games (takes ~10 seconds on a laptop CPU)
        # args.render_mode = "human"
        # args.num_games_eval = 2
        # eval(args)
    elif TRAIN_ID == 'm':
        run_dir = "run6"
        args.load_path = './models/{}/{}/{}_{}'.format(args.scenario_name, run_dir,
                                                       args.scenario_name, args.algorithm)  # 没有添加steps
        from runner_pettingzoo.ppo.configs.config_for_log import operate_config
        operate_config(args, run_dir)
        # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
        args.render_mode = None
        args.num_games_eval = 2  # 10
        eval(args)
