"""
run `pip install "gymnasium[toy-text]"` to install gymnasium
"""
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv, generate_random_map
import numpy as np
import re
import copy

from ragen.utils import NoLoggerWarnings
from ragen.env.base import BaseDiscreteActionEnv

class FrozenLakeEnv(BaseDiscreteActionEnv, GymFrozenLakeEnv):
    """
    Inherits from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv

    ## Description
    The game starts with the player at location [0,0] of the frozen lake grid world with the
    goal located at far extent of the world e.g. [3,3] for the 4x4 environment.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.
    NOTE the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
    - 0: Still
    - 1: Left
    - 2: Down
    - 3: Right
    - 4: Up

    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]).

    ## Rewards
    NOTE added -0.1 as penalty for invalid action
    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Arguments
    `is_slippery`: if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3

    ## Example
    P   _   _   _
    _   _   _   O
    O   _   O   _
    O   _   _   G
    """

    # Map gym state in integer
    MAP_LOOKUP = {
        b"P": 0,
        b"F": 1,
        b"H": 2,
        b"G": 3,
    }

    # Define rules to transform to rendered text observation of the environment
    GRID_LOOKUP = {
        0: " P \t",  # player
        1: " _ \t",  # frozen
        2: " O \t",  # hole
        3: " G \t",  # goal
        4: " X \t",  # player fall into hole
        5: " √ \t",  # player on goal
    }

    ACTION_LOOKUP = {
        0: "none",
        1: "left",
        2: "down",
        3: "right",
        4: "up",
    }

    INVALID_ACTION = 0
    ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)


    def __init__(self, **kwargs):
        BaseDiscreteActionEnv.__init__(self)

        desc = kwargs.pop('desc', None)
        is_slippery = kwargs.pop('is_slippery', True)
        size = kwargs.pop('size', 8)
        p = kwargs.pop('p', 0.8)
        seed = kwargs.pop('seed', None)
        if desc is None:
            random_map = generate_random_map(size=size, p=p, seed=seed)
        else:
            random_map = np.asarray(copy.deepcopy(desc), dtype="c")
        GymFrozenLakeEnv.__init__(
            self,
            desc=random_map,
            is_slippery=is_slippery
        )
        
        self.map_kwargs = {
            "size": size,
            "p": p,
        }
        self.env_kwargs = {
            "is_slippery": is_slippery,
            "desc": copy.deepcopy(desc),
            "seed": seed,
        }
        self.action_map = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        } # map from custom Env action to action defined in FrozenLakeEnv in gymnasium

        self.reward = 0
        self.penalty_for_invalid = -0.1

    def _get_player_position(self):
        return (self.s // self.ncol, self.s % self.ncol) # (row, col)
    



    @classmethod
    def extract_action(cls, text):
        """
        Extract action from text.
        NOTE: the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
        - 0: Still (Invalid Action)
        - 1: Left
        - 2: Down
        - 3: Right
        - 4: Up
        """
        DIRECTION_MAP = {"Left": 1, "Down": 2, "Right": 3, "Up": 4}
        pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return cls.INVALID_ACTION 
        
        if match.group(2):   
            return int(match.group(2))
        elif match.group(4): 
            return DIRECTION_MAP[match.group(4).capitalize()]
        elif match.group(5): 
            return int(match.group(5))
        
        return cls.INVALID_ACTION


    def reset(
            self,
            mode='tiny_rgb_array',
            reset_map=True,
            seed=None
    ):
        """
        Reset the environment, there are two options:
        1. reset the map, generate a new map (reset_map=True)
        2. reset the environment with the same map, while putting the agent back to the start position (reset_map=False)
        Both can reset the seed
        NOTE if seed is the same, the map will be the same
        """
        
        if reset_map:
            self.__init__(
                size=self.map_kwargs["size"],
                p=self.map_kwargs["p"],
                seed=seed,
                is_slippery=self.env_kwargs["is_slippery"],
            )
        GymFrozenLakeEnv.reset(self, seed=seed)
        
        return self.render(mode)
        
    def success(self):
        """
        Check if the agent has reached the goal (G) or hole (H)
        """
        row, col = self.s // self.ncol, self.s % self.ncol
        return self.desc[row, col] in b"GH"
    
    def step(self, action: int):
        """
        Map custom action to gymnasium FrozenLakeEnv action and take the step
        """
        assert isinstance(action, int), "Action must be an integer"
        assert not self.success(), "Agent has already reached the goal or hole"

        if action == self.INVALID_ACTION:
            return self.render(), 0, False, {}
        with NoLoggerWarnings():
            player_pos, reward, done, _, prob = GymFrozenLakeEnv.step(self, self.action_map[action])
            
        obs = self.render()
        return obs, reward, done, _
    

     
    def render(self, mode='tiny_rgb_array'):
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array', 'ansi']
        if mode in ['rgb_array', 'ansi']:
            prev_render_mode = self.render_mode
            self.render_mode = mode
            obs = GymFrozenLakeEnv.render(self)
            self.render_mode = prev_render_mode
            return obs
        room_state = copy.deepcopy(self.desc)

        # replace the position of start 'S' with 'F'
        position_S = np.where(room_state == b'S')
        room_state[position_S] = b'F'

        # replace the position of the player with 'P'
        position_P = self._get_player_position()
        room_state[position_P] = b'P'

        if mode == 'state':
            # transform 'S', 'F', 'H', 'G' to numpy integer array
            room_state = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room_state)
            # add player in hole or player on goal
            if self.desc[position_P] == b'H':
                room_state[position_P] = 4
            elif self.desc[position_P] == b'G':
                room_state[position_P] = 5
            return room_state
        
        room_state = self.render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'tiny_rgb_array':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
    
        
    
    def copy(self):
        if self.env_kwargs['seed'] is None:
            print("Warning: seed is None, copy will not be deterministic")

        new_self = FrozenLakeEnv(
            size=self.map_kwargs["size"],
            p=self.map_kwargs["p"],
            seed=self.env_kwargs["seed"],
            is_slippery=self.env_kwargs["is_slippery"],
            desc=copy.deepcopy(self.env_kwargs["desc"])
        )
        if hasattr(self, 's'):
            new_self.s = self.s
        if hasattr(self, 'lastaction'):
            new_self.lastaction = self.lastaction
        new_self.reward = self.reward
        return new_self
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def save_render_to_png(np_img, filename):
        
        fig = plt.figure(frameon=False)
        fig.set_size_inches(np_img.shape[1] / 100, np_img.shape[0] / 100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(np_img)
        plt.savefig(filename, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    env = FrozenLakeEnv(size=3, seed=5, is_slippery=False)
    env.reset(seed=4, reset_map=True)
    save_render_to_png(env.render(mode='rgb_array'), 'frozen_lake_0.png')
    print(env.render(mode='tiny_rgb_array'))
    env.step(3)
    save_render_to_png(env.render(mode='rgb_array'), 'frozen_lake_1.png')
    env.step(2)
    save_render_to_png(env.render(mode='rgb_array'), 'frozen_lake_2.png')
    env.step(3)
    save_render_to_png(env.render(mode='rgb_array'), 'frozen_lake_3.png')
    env.step(2)
    save_render_to_png(env.render(mode='rgb_array'), 'frozen_lake_4.png')
    # env.step(3)
    # save_render_to_png(env.render(mode='rgb_array'), 'frozen_lake_5.png')
    # env.step(3)
    # save_render_to_png(env.render(mode='rgb_array'), 'frozen_lake_6.png')


    # obs = FrozenLakeEnv.execute_predictions([env], ["<answer>right</answer>"], "<PAD>")
    # print(obs)
    # print(env.render(mode='tiny_rgb_array'))

    
    
    # print(env.render(mode='ansi'))