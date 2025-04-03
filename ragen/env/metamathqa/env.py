import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import re
import random
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed
from .config import MetaMathQAEnvConfig
class MetaMathQAEnv(BaseLanguageBasedEnv):
    def __init__(self, config: MetaMathQAEnvConfig):
        super(MetaMathQAEnv, self).__init__()
        
        self.config = config
        self.dataset = load_dataset(path=self.config.dataset_path, cache_dir=self.config.cache_dir)
        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        
        
    def _extract_answer(self, response):
        match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
        
    def reset(self,seed=None):
        dataset = self.dataset[self.config.split]
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        question_data = dataset[self.current_question_idx]
        self.current_question = question_data['original_question']
        self.correct_answer = self._extract_answer(question_data['response'])
        
        
        return self.current_question
        
    def step(self, action):
        is_correct = self._check_answer(action)
        reward = 1.0 if is_correct else 0.0
        if is_correct:
            observation = "Correct!"
            done = True
        else:
            observation = "Incorrect. Please think again."
            done = False
        return observation, reward, done, {}
    
    def _check_answer(self, user_answer):
        """Check if the user's answer matches the correct answer."""
        user_answer = user_answer.strip()
        normalized_user = re.sub(r'\s+', '', user_answer.lower())
        if self.correct_answer:
            normalized_correct = re.sub(r'\s+', '', self.correct_answer.lower())
            return normalized_user == normalized_correct
        return False


if __name__ == "__main__":
    # Create the environment configuration
    config = MetaMathQAEnvConfig(
        dataset_path="meta-math/MetaMathQA",
        cache_dir="./data",
        split="train"
    )
    
    # Initialize the environment
    env = MetaMathQAEnv(config)
    
    # Reset the environment to get the first question
    print("Question:")
    question = env.reset(seed=42)
    print(question)
    print("\nCorrect answer (for testing purposes):")
    print(env.correct_answer)
    
    # Interactive loop for testing
    while True:
        user_answer = input("\nEnter your answer (or 'q' to quit): ")
        if user_answer.lower() == 'q':
            break
        
        # Take a step in the environment with the user's answer
        obs, reward, done, info = env.step(user_answer)
        
        # Print the results
        print("\nFeedback:", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        
        # If the episode is done, reset the environment for a new question
        if done:
            print("\n--- New Question ---")
            question = env.reset()
            print(question)
            print("\nCorrect answer (for testing purposes):")
            print(env.correct_answer)