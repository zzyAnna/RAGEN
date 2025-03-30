from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass

@dataclass
class AlfredEnvConfig(BaseEnvConfig):
    env_type: str = "AlfredTWEnv"
    train_path: str = "data/alfworld/train.parquet"
    max_instances: int = 20000
    
    ...