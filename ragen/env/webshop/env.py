from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.webshop.config import WebShopEnvConfig
from webshop_minimal import WebAgentTextEnv
from typing import Optional, Union
from ragen.utils import all_seed
import random
import string

class WebShopEnv(BaseLanguageBasedEnv, WebAgentTextEnv):
    def __init__(self, config: Optional[WebShopEnvConfig] = None, **kwargs: any) -> None:
        """
        Adapter for WebAgentTextEnv to conform to the BaseLanguageBasedEnv interface.
        """
        self.config = config or WebShopEnvConfig()
        self.observation_mode = self.config.observation_mode
        self.file_path = self.config.file_path
        self.server = self.config.server
        self.filter_goals = self.config.filter_goals
        self.limit_goals = self.config.limit_goals
        self.num_products = self.config.num_products
        self.human_goals = self.config.human_goals
        self.show_attrs = self.config.show_attrs

        BaseLanguageBasedEnv.__init__(self)
        WebAgentTextEnv.__init__(
            self,
            observation_mode=self.observation_mode,
            file_path=self.file_path,
            server=self.server,
            filter_goals=self.filter_goals,
            limit_goals=self.limit_goals,
            num_products=self.num_products,
            human_goals=self.human_goals,
            show_attrs=self.show_attrs,
            **kwargs
        )

    def reset(self, seed=None, session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None) -> any:
        """
        Reset the environment and return the initial observation.

        Args:
            session (str|int|None): The new session ID.
            instruction_text (str|None): Optional new instruction text.

        Returns:
            The initial observation.
        """
        if session is None:
            with all_seed(seed):
                session = ''.join(random.choices(string.ascii_lowercase, k=10))
        obs, _ = WebAgentTextEnv.reset(self, session=session, instruction_text=instruction_text)
        return obs

    def step(self, action):
        """
        Take an action in the environment and return the next observation, reward, done, and info.
        """
        state, reward, done, info = WebAgentTextEnv.step(self, action)
        return self.observation, reward, done, info

    def render(self, mode=None):
        """
        Render the environment.
        """
        return WebAgentTextEnv.render(self, mode=mode)

    def close(self):
        """
        Close the environment.
        """
        WebAgentTextEnv.close(self)

if __name__ == '__main__':
    env = WebShopEnv()
    print(env.reset())
    while True:
        print(f"Available actions: {env.get_available_actions()}")
        action = input("Enter action: ")
        if action == 'q':
            break
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    env.close()
