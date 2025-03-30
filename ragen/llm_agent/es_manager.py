from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS

@dataclass
class EnvStatus:
    """Status of an environment"""
    Truncated: bool = False # done but not success
    Terminated: bool = False # done and success
    Cur_step: int = 0 # current action step (single action)
    Seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config):
        self.config = config
        self._init_envs()

    def _init_env_from_config(self, Env_config: Dict):
        """Initialize the environments from config"""
        env_type = env_config["Env_type"]
        env_config_dict = env_config["Env_config"]
        env_config = REGISTERED_ENV_CONFIGS[env_type](**env_config_dict)
        env = REGISTERED_ENVS[env_type](env_config)
        return env

    def _init_envs(self):
        """Initialize the environments from config
        config (config -> agent_proxy -> Envs): 
            Envs -> train -> env_groups (number of groups, e.g. 8)
            Envs -> train -> group_size (number of envs in each group, all envs are the same, e.g. 16)
            Envs -> train -> Env_configs:
                Tags (List[str]): tags for the envs, e.g. ["bandit", "countdown"]
                N_groups (List[int]): number of groups for each tag (sum = env_groups)
            Envs -> val

        config (config -> custom_envs): 
            custom_envs -> SimpleSokoban -> Env_type (str): "sokoban"
            custom_envs -> SimpleSokoban -> Env_config (dict): used for initilizing the env
            custom_envs -> SimpleSokoban -> Env_instruction (str): initial instructions for the env
            custom_envs -> SimpleFrozenLake ...
        """

        # Initialize training environments
        self.train_envs = []
        
        env_groups = self.config.agent_proxy.Envs.train.env_groups
        group_size = self.config.agent_proxy.Envs.train.group_size
        env_tags = self.config.agent_proxy.Envs.train.Env_configs.Tags
        n_groups = self.config.agent_proxy.Envs.train.Env_configs.N_groups
        
        # Verify that the sum of N_groups equals env_groups
        assert sum(n_groups) == env_groups, "Sum of N_groups must equal env_groups"

        Env_config_list = []
        group_idx, env_idx = 0, 0
        for tag, n_group in zip(env_tags, n_groups):
            for _ in range(n_group):
                env_config = self._init_env_from_config(self.config.custom_envs[tag])
                Env_config_list.append(env_config)
                env_idx += 1
            group_idx += 1




        
        





    def update_env_status(self, env_status: EnvStatus):
        self.env_status = env_status