class BaseConfig:
    def __init__(self) -> None:
        self.env_name = 'pettingzoo' 
        self.scenario_name = 'knights_archers_zombies_v10'
        self.experiment_name = 'norm'

        self.algorithm = 'ippo'  
        self.max_train_steps = int(1e6)
        self.episode_limit = 200
        self.evaluate_freq = 5000
        self.evaluate_times = 3
        self.seed = 0

        # eval
        self.num_games_eval = 8
        self.render_mode = None

        self.tensorboard_log = 'tensorboard/{}/{}_{}'.format(self.algorithm, self.scenario_name, self.algorithm)
        self.save_path = './model/{}/{}_{}'.format(self.algorithm, self.scenario_name, self.algorithm)
        self.load_path = self.save_path  # default
