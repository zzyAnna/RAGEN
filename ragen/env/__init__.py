from .alfworld.config import AlfredEnvConfig
from .alfworld.env import AlfredTXTEnv
from .bandit.config import BanditEnvConfig
from .bandit.env import BanditEnv
from .countdown.config import CountdownEnvConfig
from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    'alfworld': AlfredTXTEnv,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    'alfworld': AlfredEnvConfig,
}