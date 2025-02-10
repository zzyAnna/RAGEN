"""
run `pip install gym-bandits` to install the bandit environment
"""

import gymnasium as gym
import numpy as np
from typing import Optional
import copy
from ragen.utils import NoLoggerWarnings

class BanditEnv(gym.Env):
    """
    Wrapper for the BanditTenArmedGaussian environment to match the project structure.
    
    ## Description
    10 armed bandit mentioned in Sutton and Barto's RL book. Each action
    represents pulling one of 10 arms, with rewards drawn from a normal distribution.

    ## Action Space
    The action shape is `(1,)` in the range `{1, 10}` indicating which arm to pull.
    - 1-10: Pull corresponding arm

    ## Rewards
    - Mean of payout is pulled from N(0, 1)
    - Actual reward is drawn from N(mean, 1)
    """
    def __init__(
            self,
            n_arms: int = 10,
            seed: Optional[int] = None,
    ):
        self.n_arms = n_arms
        self.action_space = gym.spaces.discrete.Discrete(n_arms, start=1)
        
        # Initialize reward distributions for each arm
        self.reward_means = None
        self.env_kwargs = {
            "n_arms": n_arms,
            "seed": seed,
        }
        
        # Initialize tracking variables
        self.last_action = None
        self.reward = 0
        self.reset(seed=seed)

    def reset(self, seed=None, mode='text'):
        """Reset the environment and reward distributions"""
        if seed is not None:
            np.random.seed(seed)
            
        # Generate new reward distributions
        self.reward_means = np.random.normal(0, 1, size=self.n_arms)
        self.reward = 0
        self.last_action = None
        
        return self.render(mode)

    def step(self, action: int):
        """Take action (pull arm) and get reward"""
        assert isinstance(action, int)
        assert 1 <= action <= self.n_arms, f"Invalid action {action}"
        
        arm_idx = action - 1  # Convert to 0-based index
        reward = np.random.normal(self.reward_means[arm_idx], 1)
        self.last_action = action
        self.reward += reward
        
        return self.render(), reward, True, {}

    def render(self, mode='text'):
        """Render the current state"""
        if mode == 'text':
            if self.last_action is None:
                return "You are facing 10 slot machines (numbered 1-10). Each machine has its own reward distribution.\nWhich machine will you pull?"
            else:
                return f"You pulled machine {self.last_action}.\nWhich machine will you pull next?"
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    @staticmethod
    def parse_update_info_to_obs(update_info, action_is_valid):
        observation, reward, done, _ = update_info
        if not action_is_valid:
            output_str = f"Invalid action. Please choose a number between 1 and 10.\nThe observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        else:
            output_str = f"After pulling the arm, the observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        return output_str

    @classmethod
    def execute_predictions(cls, envs, predictions, pad_token):
        """Execute predictions across multiple environments"""
        cur_actions, action_is_valid = cls.postprocess_predictions(predictions)
        next_obs = []
        dones = []
        
        for env, action, response, av in zip(envs, cur_actions, predictions, action_is_valid):
            obs = ""
            if "<|im_end|>" not in response:
                obs += "<|im_end|>"
                
            observation, reward, done, extra_info = env.step(action)
            env_feedback = cls.parse_update_info_to_obs((observation, reward, done, extra_info), av)
            obs += "\n <|im_start|>user\n" + env_feedback + "<|im_end|>\n" + "<|im_start|>assistant\n<think>"
            dones.append(done)
            next_obs.append(obs)
            
        return next_obs, dones

    @staticmethod
    def extract_action(text):
        """Extract action number from text response"""
        import re
        pattern = r'^\s*([1-9]|10)\s*$'
        match = re.search(pattern, text.strip())
        return int(match.group(1)) if match else 0

    @staticmethod
    def postprocess_predictions(predictions):
        """Process predictions into actions"""
        actions = []
        action_is_valid = []
        
        for prediction in predictions:
            if isinstance(prediction, str):
                if "<answer>" in prediction:
                    action_text = prediction.split("<answer>")[1].split("</answer>")[0].strip()
                else:
                    action_text = prediction.strip()
                    
                action = BanditEnv.extract_action(action_text)
                action_is_valid.append(action != 0)
                
            elif isinstance(prediction, (int, float)):
                action = int(prediction)
                action_is_valid.append(1 <= action <= 10)
            else:
                action = 0
                action_is_valid.append(False)
                
            actions.append(action)
            
        return actions, action_is_valid

    def get_all_actions(self):
        """Return all valid actions"""
        return list(range(self.action_space.start, self.action_space.start + self.action_space.n))

    def copy(self):
        """Create a copy of the environment"""
        new_env = BanditEnv(
            n_arms=self.env_kwargs["n_arms"],
            seed=self.env_kwargs["seed"]
        )
        new_env.reward_means = self.reward_means.copy()
        new_env.last_action = self.last_action
        new_env.reward = self.reward
        return new_env