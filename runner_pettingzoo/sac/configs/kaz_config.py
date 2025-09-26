from .base_config import BaseConfig


class KAZConfig(BaseConfig):
    def __init__(self, scenario_name, algorithm) -> None:
        super().__init__()
        self.env_name = 'pettingzoo' 
        self.scenario_name = scenario_name
        self.experiment_name = 'norm'

        self.algorithm = algorithm  
        self.max_train_steps = int(100)
        self.episode_limit = 900  # kaz默认为900
        self.evaluate_freq = 5000
        self.evaluate_times = 3
        self.seed = 0

        # eval
        self.num_games_eval = 8
        self.render_mode = 'human'

        self.tensorboard_log = 'tensorboard/{}/{}_{}'.format(self.algorithm, self.scenario_name, self.algorithm)
        self.save_path = './model/{}/{}_{}'.format(self.algorithm, self.scenario_name, self.algorithm)
        self.load_path = self.save_path  # default

        # env
        self.max_zombies = 10  # kaz默认为10
        self.vector_state = True
