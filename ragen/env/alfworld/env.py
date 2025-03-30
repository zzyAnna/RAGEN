import gym
import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from .config import AlfredEnvConfig
from .utils import load_config
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from ragen.env.base import BaseDiscreteActionEnv

class AlfredEnv(BaseDiscreteActionEnv, AlfredTWEnv):
    def __init__(self, config: AlfredEnvConfig = AlfredEnvConfig(), **kwargs):
        self.config = config
        self.ACTION_LOOKUP = self.config.action_lookup
        self.ACTION_HELP_LOOKUP = self.config.action_help_lookup
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(len(self.ACTION_LOOKUP), start=1)

        BaseDiscreteActionEnv.__init__(self)

if __name__ == "__main__":
    ...