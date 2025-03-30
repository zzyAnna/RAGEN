import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from .config import AlfredEnvConfig
from .utils import load_config
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv



from ragen.env.base import BaseLanguageBasedEnv

class AlfredEnv(BaseLanguageBasedEnv, AlfredTWEnv):
    def __init__(self, config: AlfredEnvConfig = AlfredEnvConfig()):
        self.config = config
        ...


if __name__ == "__main__":
    ...