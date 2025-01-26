from .base import BasePolicy
import random

class FixedPolicy(BasePolicy):
    def select_action(self, observation, env=None):
        return random.choice(env.get_all_actions())

class HeuristicPolicy(BasePolicy):
    """ A heuristic policy that moves towards the closest box/target for SokobanEnv. """
    def select_action(self, observation, env=None):
        obs = [row.strip() for row in observation.strip().split("\n")]
        player, targets = None, []

        for i, row in enumerate(obs):
            for j, cell in enumerate(row):
                if cell in ("P", "S"): player = (i, j)
                if cell in ("X", "O"): targets.append((i, j))

        if not player or not targets:
            return random.choice(range(1, 5))

        px, py = player
        tx, ty = min(targets, key=lambda t: abs(px - t[0]) + abs(py - t[1]))
        moves = [1 if tx < px else 2 if tx > px else None, 
                 3 if ty < py else 4 if ty > py else None]
        return random.choice([m for m in moves if m])
