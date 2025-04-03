import gymnasium as gym
import numpy as np
from ragen.env.base import BaseDiscreteActionEnv
from .config import BanditEnvConfig

class BanditEnv(BaseDiscreteActionEnv, gym.Env):
    def __init__(self, config = None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config if config is not None else BanditEnvConfig()
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=self.config.action_space_start)
        self.lo_arm_name = self.config.lo_arm_name
        self.hi_arm_name = self.config.hi_arm_name
        self.render_cache = None
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        
    def _randomize_arms(self):
        start = self.config.action_space_start
        if self.np_random.random() < 0.5:
            self.ACTION_LOOKUP = {
                start: self.lo_arm_name,
                start + 1: self.hi_arm_name,
            }
        else:
            self.ACTION_LOOKUP = {
                start: self.hi_arm_name,
                start + 1: self.lo_arm_name,
            }
        self.config.action_lookup = self.ACTION_LOOKUP
        self.ARM_IDX_TO_NAME = self.ACTION_LOOKUP
        self.NAME_TO_ARM_IDX = {name: idx for idx, name in self.ACTION_LOOKUP.items()}

    def _lo_arm_reward(self):
        return self.config.lo_arm_score

    def _hi_arm_reward(self):
        if self.np_random.random() < self.config.hi_arm_hiscore_prob:
            return self.config.hi_arm_hiscore
        return self.config.hi_arm_loscore

    def render(self):
        return self.render_cache

    def reset(self, seed=None):
        gym.Env.reset(self, seed=seed)
        self._randomize_arms()
        pos1 = self.config.action_space_start
        pos2 = pos1 + 1
        machine1 = self.ARM_IDX_TO_NAME[pos1]
        machine2 = self.ARM_IDX_TO_NAME[pos2]
        self.render_cache = f"Machines: {machine1}({pos1}), {machine2}({pos2}). Choose: {self.get_all_actions()}"
        return self.render_cache

    def step(self, action: int):
        assert action in self.ACTION_LOOKUP, f"Invalid action: {action}"
        reward = self.compute_reward(action)
        arm_name = self.ARM_IDX_TO_NAME[action]
        next_obs = f"{arm_name}: {reward} points"
        self.render_cache = next_obs
        done, info = True, {"action_is_effective": True, "action_is_valid": True, "success": arm_name == self.hi_arm_name}
        return next_obs, reward, done, info
    
    def compute_reward(self, action):
        arm_name = self.ARM_IDX_TO_NAME[action]
        if arm_name == self.lo_arm_name:
            return self._lo_arm_reward()
        else:
            return self._hi_arm_reward()

    def get_all_actions(self):
        return [self.ACTION_SPACE.start, self.ACTION_SPACE.start + 1]

    def render(self):
        return self.render_cache

    def close(self):
        self.render_cache = None

if __name__ == "__main__":
    def run_simulation(env, n_episodes=1000, action=1, start_seed=500):
        rewards = []
        for i in range(start_seed, start_seed + n_episodes):
            env.reset(seed=i)
            reward = env.step(action)[1]
            rewards.append(reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'n_episodes': n_episodes,
            'action': env.ARM_IDX_TO_NAME[action]
        }

    env = BanditEnv()
    stats = run_simulation(env)
    print(f"Arm: {stats['action']}, Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")