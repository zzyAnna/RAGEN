from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import PIL.Image
import hydra

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    cur_step: int = 0 # current action step (single action)
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config):
        self.config = config
        self.env_groups = config.es_manager.train.env_groups
        self.group_size = config.es_manager.train.group_size
        self._init_envs()

    def _init_env_from_config(self, env_config: Dict):
        """Initialize the environments from config"""
        env_type = env_config["env_type"]
        env_config_dict = env_config["env_config"]
        env_config = REGISTERED_ENV_CONFIGS[env_type](**env_config_dict)
        env = REGISTERED_ENVS[env_type](env_config)
        return env

    def _init_env_list(self, config):
        """Initialize the environment list
        Tags: ["SimpleSokoban", "HarderSokoban"]
        n_groups: [1, 2]
        group_size: 16

        env_list = [
            {"tag": "SimpleSokoban", "group_idx": 0, "env_id": 0, "env": env, "env_config": env_config, "status": EnvStatus()}
            {"tag": "SimpleSokoban", "group_idx": 0, "env_id": 1, ...}
            ...
            {"tag": "SimpleSokoban", "group_idx": 0, "env_id": 15 (group_size - 1), ...}
            {"tag": "HarderSokoban", "group_idx": 1, "env_id": 16, ...}
            ...
            {"tag": "HarderSokoban", "group_idx": 1, "env_id": 31, ...}
            {"tag": "HarderSokoban", "group_idx": 2, "env_id": 32, ...}
        ]
        """

        group_idx, env_idx = 0, 0
        env_list = []
        for tag, n_group in zip(config.env_configs.Tags, config.env_configs.n_groups):
            for _ in range(n_group):
                for _ in range(config.group_size):
                    env_config = self.config.custom_envs[tag]
                    env = self._init_env_from_config(env_config)
                    env_entry = {
                        'tag': tag,
                        'group_idx': group_idx,
                        'env_id': env_idx,
                        'env': env,
                        'env_config': env_config,
                        'status': EnvStatus()
                    }
                    env_list.append(env_entry)
            group_idx += 1
        return env_list

    def _init_envs(self):
        """Initialize the environments from config
        config (config -> agent_proxy -> envs): 
            envs -> train -> env_groups (number of groups, e.g. 8)
            envs -> train -> group_size (number of envs in each group, all envs are the same, e.g. 16)
            envs -> train -> env_configs:
                Tags (List[str]): tags for the envs, e.g. ["bandit", "countdown"]
                n_groups (List[int]): number of groups for each tag (sum = env_groups)
            envs -> val

        config (config -> custom_envs): 
            custom_envs -> SimpleSokoban -> Env_type (str): "sokoban"
            custom_envs -> SimpleSokoban -> env_config (dict): used for initilizing the env
            custom_envs -> SimpleSokoban -> Env_instruction (str): initial instructions for the env
            custom_envs -> SimpleFrozenLake ...
        """

        train_env_config = self.config.es_manager.train
        val_env_config = self.config.es_manager.val
        if not val_env_config:
            val_env_config = train_env_config
        
        # Verify that the sum of n_groups equals env_groups
        assert sum(train_env_config.env_configs.n_groups) == train_env_config.env_groups, "Sum of n_groups must equal env_groups"

        self.train_envs = self._init_env_list(train_env_config)
        self.val_envs = self._init_env_list(val_env_config)
        self.log = []


    def _get_history_entry(
            self,
            state,
            reward: Optional[float] = None,
            actions: Optional[List[str]] = None,
            info: Optional[Dict] = None,
            llm_response: Optional[str] = None,
        ):
        """
        Entry for an environment rollout history
        """
        entry = {'state': ""}
        if isinstance(state, list):
            entry['images'] = []
            for _state in state:
                entry['state'] += f"<image>"
                entry['images'].append(_state)
        else:
            entry['state'] = state
        if reward is not None:
            entry['reward'] = reward
        if actions is not None:
            entry['actions'] = actions
        if info is not None:
            entry['info'] = info
        if llm_response is not None:
            entry['llm_response'] = llm_response
        return entry




    def reset(
            self,
            val: bool = False,
            seed = None
        ):
        """Reset the environments and get initial observation

        Args:
            val (bool): whether to reset the validation environments
            seed (List[int] | int): seeds for the environments

        """

        def _expand_seed(seed):
            if isinstance(seed, list):
                assert len(seed) == self.env_groups, "Seed list length must equal env_groups"
                # expand seed to group_size
                seeds = []
                for _seed in seed:
                    seeds.extend([_seed] * self.group_size)
                return seeds
            elif isinstance(seed, int):
                seeds = []
                for _ in range(self.env_groups):
                    seeds.extend([seed] * self.group_size)
                    seed += 1
                return seeds
            else:
                raise ValueError(f"Invalid seed type: {type(seed)}")

        envs = self.val_envs if val else self.train_envs
        log = [{"env_id": entry['env_id'], "history": {}} for entry in envs]

        seeds = _expand_seed(seed)
        for env_id, env in enumerate(envs):
            env['env'].reset(seed=seeds[env_id])
            obs = env['env'].render()
            log[env_id]['history'].append(self._get_history_entry(state=obs))

        self.log = log
        return log

    def step(
            self,
            all_env_inputs: List[Dict],
            val: bool = False,
        ):
        """Step the environments

        Args:
            all_env_inputs (List[Dict]): inputs for all environments
                each entry: {env_id: int, llm_response: str, parsed_response: List[str]}
            val (bool): whether to step the validation environments

        Returns:
            env_outputs (List[Dict]): outputs for all environments
                each entry (log): {env_id: int, history: List[Dict]}
        """
        env_outputs = []

        envs = self.val_envs if val else self.train_envs
        log = [{"env_id": entry['env_id'], "history": {}} for entry in envs]

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            env_id = env_input['env_id']
            env_entry = envs[env_id]
            env = env_entry['env']
            for action in env_input['parsed_response']:
                _, reward, done, info = env.step(action)
                
                acc_reward += reward
                turn_info.update(info)
                if done:
                    turn_done = True
                    break
            obs = env.render()
            if turn_done: # TODO: check if this is correct
                env_entry['status'].terminated = env.success()
                env_entry['status'].truncated = not env.success()
            history_entry = self._get_history_entry(
                state=obs,
                llm_response=env_input['llm_response'],
                actions=env_input['parsed_response'],
                reward=acc_reward,
                info=turn_info
            )
            log[env_id]['history'].append(history_entry)
            if not turn_done:
                env_outputs.append(log[env_id])



        
    def render(self, val: bool = False):
        envs = self.val_envs if val else self.train_envs
        render_list = []
        for env in envs:
            render_list.append(env['env'].render())
        return render_list

    def close(self, val: bool = False):
        envs = self.val_envs if val else self.train_envs
        for env in envs:
            env['env'].close()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    es_manager = EnvStateManager(config)
    es_manager.reset()
    es_manager.step()
    es_manager.render()
    es_manager.close()


if __name__ == "__main__":
	main()
