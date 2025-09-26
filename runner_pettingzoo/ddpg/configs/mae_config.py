from .base_config import BaseConfig


class MAEConfig(BaseConfig):
    def __init__(self, scenario_name, algorithm) -> None:
        super().__init__()
        self.env_name = 'MAE' 
        self.scenario_name = scenario_name
        self.experiment_name = 'norm'

        self.algorithm = algorithm  
        self.max_train_steps = int(1e6)
        self.episode_limit = 200
        self.evaluate_freq = 5000
        self.evaluate_times = 3
        self.seed = 123

        # eval
        self.num_games_eval = 8
        self.render_mode = None

        self.tensorboard_log = 'tensorboard/{}/{}_{}'.format(self.algorithm, self.scenario_name, self.algorithm)
        self.save_path = './model/{}/{}_{}'.format(self.algorithm, self.scenario_name, self.algorithm)
        self.load_path = self.save_path  # default

        self.num_agents = 2
        self.continuous_actions = False
        self.scr_mode = 'mini_max'
        self.shared_reward = False
        self.tar_mode = 'auction'  # auction;distance
        self.simu_init = [[0, 0, 3000, 45, 0, 200],
                          [1000, 0, 3000, 45, 0, 200],
                          [3000, 3000, 3000, 90, 0, 220],
                          [4000, 4000, 3000, 90, 0, 220]]
        self.use_seed = True
        self.en = True
        self.only_fig = False
        self.sub_fig = False
        self.selfplay = False
        self.curriculum_learning = False
        self.transfer_learning = False

        # sac
        self.gamma = 0.99
