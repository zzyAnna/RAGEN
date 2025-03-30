"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
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
        self.env_groups = int(config.es_manager.train.env_groups)
        self.group_size = int(eval(config.es_manager.train.group_size)) # TODO: change rollout_n in base.yaml 
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
                for _ in range(self.group_size):
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
        assert sum(train_env_config.env_configs.n_groups) == self.env_groups, "Sum of n_groups must equal env_groups"

        self.train_envs = self._init_env_list(train_env_config)
        self.val_envs = self._init_env_list(val_env_config)
        self.rollout_cache = [] # cache for rollout logs for each env

    def _update_history(
            self,
            history: List[Dict],
            next_state,
            last_step_info: Optional[Dict] = None,
        ):
        """
        Entry for an environment rollout history
        """
        if last_step_info is not None:
            assert len(history), "History should not be empty"
            history[-1].update(last_step_info)

        entry = {'state': ""}
        if isinstance(next_state, list):
            entry['images'] = []
            for _state in next_state:
                entry['state'] += f"<image>"
                entry['images'].append(_state)
        else:
            entry['state'] = next_state
        history.append(entry)

    def _parse_env_input(self, env_entry: Dict, parsed_llm_response: List[str]):
        """Parse the LLM response for the environment
        """
        valid_actions = []
        action_lookup = env_entry['env_config'].str_to_action_lookup
        if action_lookup is not None:
            for action in parsed_llm_response:
                if action in action_lookup:
                    valid_actions.append(action_lookup[action])
        else:
            valid_actions = parsed_llm_response
        
        return valid_actions


    def reset(
            self,
            val: bool = False,
            seed = None
        ):
        """Reset the environments and get initial observation

        Args:
            val (bool): whether to reset the validation environments
            seed (List[int] | int): seeds for the environments

        Returns:
            rollout_cache (List[Dict]): cache for rollout logs for each env
                each entry: {env_id: int, history: List[Dict]}
        """
        envs = self.val_envs if val else self.train_envs
        rollout_cache = [{"env_id": entry['env_id'], "history": []} for entry in envs]


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
        seeds = _expand_seed(seed)


        for env_id, env in enumerate(envs):
            env['env'].reset(seed=seeds[env_id])
            obs = env['env'].render()
            self._update_history(rollout_cache[env_id]['history'], obs)
            env['status'] = EnvStatus(seed=seeds[env_id])

        self.rollout_cache = rollout_cache
        return rollout_cache

    def step(
            self,
            all_env_inputs: List[Dict],
            val: bool = False,
        ):
        """Step the environments

        Args:
            all_env_inputs (List[Dict]): inputs for all environments
                each entry: {env_id: int, llm_response: str, actions: List[str]}
                NOTE: should use env_id as index for existing some already done envs
            val (bool): whether to step the validation environments

        Returns:
            env_outputs (List[Dict]): outputs for not done environments
                each (rollout_cache) entry: {env_id: int, history: List[Dict]}
        """
        env_outputs = []

        envs = self.val_envs if val else self.train_envs

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            env_id = env_input['env_id']
            env_entry = envs[env_id]
            env = env_entry['env']
            valid_actions = self._parse_env_input(env_entry, env_input['actions'])
            executed_actions = []
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                acc_reward += self.config.es_manager.format_penalty
            for action in valid_actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            obs = env.render()
            env_entry['status'].rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            env_entry['status'].cur_step += len(executed_actions)
            if turn_done:
                env_entry['status'].terminated = True # TODO check terminated definition in gymnasium
                env_entry['status'].truncated = not env.success()
            self._update_history(
                self.rollout_cache[env_id]['history'],
                next_state=obs,
                last_step_info={
                    'actions': executed_actions,
                    'reward': acc_reward,
                    'info': turn_info,
                    'llm_response': env_input['llm_response'],
                    'llm_raw_response': env_input['llm_raw_response']
                }
            )
            if not turn_done: # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def render_final_output(self, val: bool = False):
        """Get the final output for all environment

        Returns:
            final_output (List[Dict]): outputs for not done environments
                each entry: {env_id: int, history: List[Dict], group_idx: int}
        """
        envs = self.val_envs if val else self.train_envs
        final_output = []
        for cache_entry in self.rollout_cache:
            env_id = cache_entry['env_id']
            final_output.append({
                'env_id': env_id,
                'history': cache_entry['history'],
                'group_idx': envs[env_id]['group_idx'],
            })
        return final_output

        
    def render(self, val: bool = False):
        envs = self.val_envs if val else self.train_envs
        render_list = []
        for env_entry in envs:
            render_list.append(env_entry['env'].render())
        return render_list

    def close(self, val: bool = False):
        envs = self.val_envs if val else self.train_envs
        for env_entry in envs:
            env_entry['env'].close()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config)
    print("Initializing environments...")
    es_manager.reset(seed=123)
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs, val=False)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    print("\nRendering training environments...")
    renders = es_manager.render(val=False)
    for i, render in enumerate(renders[:2]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.render_final_output(val=False)
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
	main()
