from .base import BasePolicy
from collections import deque
import pickle

class BFSPolicy(BasePolicy):
    def select_action(self, observation, env=None):
        queue = deque([(pickle.dumps(env), [])])
        visited = {env.render(mode="state").tobytes()}
        list_visited = []

        while queue:
            state, actions = queue.popleft()
            env = pickle.loads(state)

            if env.success():
                return actions if actions else 0

            for action in env.get_all_actions():
                env_copy = pickle.loads(state)
                env_copy.step(action)
                obs = env_copy.render()
                if obs not in visited:
                    visited.add(obs)
                    list_visited.append(obs)
                    queue.append((pickle.dumps(env_copy), actions + [action]))
        return None