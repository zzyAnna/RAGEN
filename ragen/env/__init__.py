from ragen.env.bandit.config import BanditEnvConfig
from ragen.env.bandit.env import BanditEnv
from ragen.env.countdown.config import CountdownEnvConfig
from ragen.env.countdown.env import CountdownEnv
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.env.sokoban.env import SokobanEnv
from ragen.env.frozen_lake.config import FrozenLakeEnvConfig
from ragen.env.frozen_lake.env import FrozenLakeEnv


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
}