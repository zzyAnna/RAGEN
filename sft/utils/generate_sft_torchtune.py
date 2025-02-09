"""
Generate SFT data for finetuning using torchtune framework
sharegpt format:
[
    {
        'dialogue': [
            {
                'from': 'human' or 'gpt',
                'value': '...',
            },
            ...
        ]
    },
    ...
]

LLM needs to output ground-truth action sequence for each environment
human needs to provide the current state of the sokoban environment after action


NOTE seed is set as 100000 to avoid overlap with training data for RAGEN
NOTE: TODO
"""

import os
import json
import torch
import numpy as np
import argparse

from ragen.env.sokoban import SokobanEnv
from ragen.policy.heuristic import FixedPolicy
from ragen.evaluators.trajectory_evaluator import TrajectoryEvaluator


qwen_instruct_template = """
<|im_start|>system
You are a helpful assistant. <|im_end|>
<|im_start|>user
{prompt}
Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra test. Strictly follow this format. <|im_end|>
<|im_start|>assistant
<think>
"""

base_template = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.
User: {prompt}
Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>
Assistant: 
<think>
"""


templates = {
    'qwen-instruct': qwen_instruct_template,
    'base': base_template
}





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="sokoban", help="Environment name (default: 'sokoban').")
    parser.add_argument("--algo", type=str, default="bfs", choices=["bfs"], help="Algorithm to use (default: 'bfs').")
    parser.add_argument("--seed", type=int, default=100000, help="Seed for random number generation (default: 100000).")
    parser.add_argument("--output", type=str, default="data/sokoban", help="Output file to save the trajectories (default: 'data/sokoban').")
    parser.add_argument("--data_size", type=int, default=1000, help="Number of trajectories to generate (default: 3000).")
    parser.add_argument("--bfs_max_nodes", type=int, default=1000, help="Maximum number of nodes to use for BFS (default: 100000).") # not using this now. This will usually give the best traj. To compare with SFT, we will try this later.
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])
    parser.add_argument('--output_file', type=str, default='data/sft_data.json')
    args = parser.parse_args()

    assert args.env == "sokoban", "Unsupported environment: {args.env}"
    assert args.algo == "bfs", "Unsupported algorithm: {args.algo}" # bfs to find shortest action path
    data_source = args.env
    
    dim_x, dim_y, num_boxes, max_steps, search_depth = os.environ.get("DIM_X"), os.environ.get("DIM_Y"), os.environ.get("NUM_BOXES"), os.environ.get("MAX_STEPS"), os.environ.get("SEARCH_DEPTH")
    dim_x, dim_y, num_boxes, max_steps, search_depth = int(dim_x), int(dim_y), int(num_boxes), int(max_steps), int(search_depth)

    env = SokobanEnv(dim_room=(dim_x, dim_y), num_boxes=num_boxes, max_steps=max_steps, search_depth=search_depth)
    policy = FixedPolicy()
    evaluator = TrajectoryEvaluator(env, policy, max_steps=1)

    # Generate trajectories
    seeds = range(args.seed, args.seed + args.data_size)
    trajectories = evaluator.batch_evaluate(seeds, mp=True) # mp=False is not working


    

if __name__ == "__main__":
    main()




