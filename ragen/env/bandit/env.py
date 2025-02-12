"""
adapted from gym-bandits
"""

import gymnasium as gym
import numpy as np
from typing import Optional
import copy
import re


from ragen.utils import NoLoggerWarnings
from ragen.env.base import BaseDiscreteActionEnv

class BanditEnv(BaseDiscreteActionEnv, gym.Env):
    """
    Wrapper for the BanditTenArmedGaussian environment to match the project structure.
    
    ## Description
    N armed bandit mentioned in Sutton and Barto's RL book. Each action
    represents pulling one of N arms, with rewards drawn from a normal distribution.

    ## Action Space
    The action shape is `(1,)` in the range `{1, N}` indicating which arm to pull.
    - 1-N: Pull corresponding arm

    ## Rewards
    - Mean of each arm is pulled from N(0, 1)
    - Actual reward of each arm is drawn from N(mean, 1)
    """

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1
    
    def __init__(
            self,
            n_arms: int = 2,
            seed: Optional[int] = None,
    ):
        self.n_arms = n_arms
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(n_arms, start=1)
        
        # Initialize reward distributions for each arm
        self.reward_means = None
        self.env_kwargs = {
            "n_arms": n_arms,
            "seed": seed,
        }
        
        # Initialize tracking variables
        self.last_action = None
        self.reward = 0

    def reset(self, seed=None, mode='text'):
        """Reset the environment and reward distributions"""
        gym.Env.reset(self, seed=seed)
            
        # Generate new reward distributions
        self.reward_means = self.np_random.normal(0, 1, size=self.n_arms)
        self.reward = 0
        self.last_action = None
        
        return self.render(mode)

    def step(self, action: int):
        """Take action (pull arm) and get reward"""
        assert isinstance(action, int)
        if action == self.INVALID_ACTION:
            return self.render(), 0, False, {}
        
        assert action in self.get_all_actions(), f"Invalid action {action}"
        
        arm_idx = action - 1  # Convert to 0-based index
        reward = self.np_random.normal(self.reward_means[arm_idx], 1)
        self.last_action = action
        
        return self.render(), reward, False, {}

    def render(self, mode='text'):
        """Render the current state"""
        if mode == 'text':
            if self.last_action is None:
                return f"You are facing {self.ACTION_SPACE.n} slot machines (numbered {self.ACTION_SPACE.start} to {self.ACTION_SPACE.start + self.ACTION_SPACE.n - 1}). Each machine has its own reward distribution.\nWhich machine will you pull?"
            else:
                return f"You pulled machine {self.last_action}.\nWhich machine will you pull next?"
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def parse_update_info_to_obs(self, update_info, action_is_valid):
        observation, reward, done, _ = update_info
        if not action_is_valid:
            output_str = f"Invalid action. Please choose a number between {self.ACTION_SPACE.start} and {self.ACTION_SPACE.start + self.ACTION_SPACE.n - 1}.\nThe observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        else:
            output_str = f"After pulling the arm, the observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        return output_str


    def extract_action(self, text):
        """Extract action number from text response"""
        
        pattern = r'^\s*(\d+)\s*$'
        match = re.search(pattern, text.strip())
        if match is None:
            return self.INVALID_ACTION
        try:
            action = int(match.group(1))
        except (ValueError, IndexError):
            print(f"Invalid action when parsing in BanditEnv: {text}")
            return self.INVALID_ACTION
        if action in self.get_all_actions():
            return action
        else:
            return self.INVALID_ACTION


    def copy(self):
        """Create a copy of the environment"""
        new_env = BanditEnv(
            n_arms=self.env_kwargs["n_arms"],
            seed=self.env_kwargs["seed"]
        )
        new_env.reward_means = copy.deepcopy(self.reward_means)
        new_env.last_action = self.last_action
        new_env.reward = self.reward
        return new_env
    
    def success(self):
        return False
    


class TwoArmedBanditEnv(BaseDiscreteActionEnv, gym.Env):
    """
    Wrapper for the BanditTenArmedGaussian environment to match the project structure.
    Two-armed bandit environment, with one phoenix arm and one dragon arm.
    Phoenix --> 1
    Dragon --> 2
    ## Action Space
    The action shape is `(1,)` in the range `{1, 2}` indicating which arm to pull.
    - 1-2: Pull corresponding arm

    ## Rewards
    - For phoenix arm (P ~ f_p), the reward is drawn from one gaussian distribution
        - f_p ~ N(0.05, 0.01)
        - mean ~ 0.05, std ~ 0.1
    - For dragon arm (D ~ f_d), the reward is drawn from a mixture of two gaussian distributions, with
        - a larger mean and std compared to the phoenix arm,
        - f_d ~ 9/10 * N(0.01, 0.01) + 1/10 * N(1, 0.01)
        - mean ~ 0.11, std ~ 0.314
    - P(P - D > 0) ~ 0.55 > 0.5

    NOTE only one step
    """

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -0.1
    ARM_IDX_TO_NAME = {
        1: "phoenix",
        2: "dragon",
    }
    NAME_TO_ARM_IDX = {
        "phoenix": 1,
        "dragon": 2,
    }
    
    def __init__(
            self,
            first_phoenix_arm: bool = True,
            seed: Optional[int] = None,
    ):
        """
        The phoenix arm is the first arm.
        If first_phoenix_arm is False, then needs to map the action space {1, 2} to {2, 1}.
        """
        self.n_arms = 2
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(self.n_arms, start=1)
        self.first_phoenix_arm = first_phoenix_arm
        
        # Initialize reward distributions for each arm

        self.env_kwargs = {
            "n_arms": 2,
            "first_phoenix_arm": first_phoenix_arm,
            "seed": seed,
        }
        
        # Initialize tracking variables
        self.last_action = None
        self.reward = 0
        self._success = False

    def _phoenix_arm_reward_distribution(self):
        """
        Distribution of phoenix arm:
            f_p ~ N(0.05, 0.01)
        """
        return self.np_random.normal(0.05, 0.1)

    def _dragon_arm_reward_distribution(self):
        """
        Distribution of dragon arm: 
            f_d ~ 9/10 * N(0.01, 0.01) + 1/10 * N(1, 0.01)
        """
        if self.np_random.random() < 0.9:
            return self.np_random.normal(0.01, 0.1)
        else:
            return self.np_random.normal(1, 0.1)
        




    def reset(self, seed=None, mode='text'):
        """Reset the environment and reward distributions"""
        gym.Env.reset(self, seed=seed)
            
        # Generate new reward distributions
        self.reward = 0
        self.last_action = None
        self._success = False
        
        return self.render(mode)

    def step(self, action: int):
        """
        Take action (pull arm) and get reward
        - action = 1: pull phoenix arm
        - action = 2: pull dragon arm
        - if not first_phoenix_arm, then map the action space {1, 2} to {2, 1}

        # TODO: if action is invalid, whether to terminate the episode?
        """
        assert isinstance(action, int)
        self._success = True
        if action == self.INVALID_ACTION:
            return self.render(), 0, True, {}
        
        assert action in self.get_all_actions(), f"Invalid action {action}"

        if not self.first_phoenix_arm: 
            # map the action space {1, 2} to {2, 1}
            action = 3 - action
        
        if action == 1:
            reward = self._phoenix_arm_reward_distribution()
        else:
            reward = self._dragon_arm_reward_distribution()
        self.last_action = action
        
        return self.render(), reward, True, {} # only one step

    def render(self, mode='text'):
        """Render the current state"""
        assert mode in ['text', 'rgb_array', 'tiny_rgb_array']
        if mode in ['text', 'tiny_rgb_array']:
            if self.last_action is None:
                return f"You are facing {self.ACTION_SPACE.n} slot machines (phoenix and dragon). Each machine has its own reward distribution.\nWhich machine will you pull?"
            else:
                return f"You pulled machine {self.ARM_IDX_TO_NAME[self.last_action]}."
        
        if mode == 'rgb_array':
            # return an empty image
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

    def parse_update_info_to_obs(self, update_info, action_is_valid):
        observation, reward, done, _ = update_info
        if not action_is_valid:
            output_str = f"Invalid action. Please choose between phoenix and dragon.\nThe observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        else:
            output_str = f"After pulling the arm, the observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        return output_str

        
    def extract_action(self, text):
        """Extract action number or specific keywords from text response"""

        pattern = r'^\s*(\d+|phoenix|dragon)\s*$'
        match = re.search(pattern, text.strip(), re.IGNORECASE)
        if match is None:
            return self.INVALID_ACTION
        try:
            # Check if the input is a number
            if match.group(1).isdigit():
                action = int(match.group(1))
                if action in self.get_all_actions():
                    return action
                else:
                    return self.INVALID_ACTION
            else:
                # Handle "phoenix" and "dragon" as valid actions
                keyword_action = match.group(1).lower()
                return self.NAME_TO_ARM_IDX[keyword_action]
        except (ValueError, IndexError):
            print(f"Invalid action when parsing in TwoArmedBanditEnv: {text}")
            return self.INVALID_ACTION



    def copy(self):
        """Create a copy of the environment"""
        new_env = TwoArmedBanditEnv(
            first_phoenix_arm=self.env_kwargs["first_phoenix_arm"],
            seed=self.env_kwargs["seed"]
        )
        new_env.last_action = self.last_action
        new_env.reward = self.reward
        return new_env
    
    def success(self):
        return self._success


if __name__ == "__main__":
    env = TwoArmedBanditEnv(first_phoenix_arm=True)
    print(env.reset(seed=0))
    print(env.step(1))

    r = []
    for i in range(500000):
        env.reset(seed=i)
        r1 = env._phoenix_arm_reward_distribution()
        r2 = env._dragon_arm_reward_distribution()
        r.append(r1 > r2)
        # r.append(r2)
    print(np.mean(r))

    # print(BanditEnv.execute_predictions([env], ["<answer>9</answer>"], "<PAD>"))