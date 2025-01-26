"""
Preprocess dataset for sokoban task - see rage/env/sokoban/sokoban.py for details
"""

import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets

from rage.policy.bfs import BFSPolicy
from rage.policy.heuristic import FixedPolicy
from rage.env.sokoban import SokobanEnv
from rage.evaluators.trajectory_evaluator import TrajectoryEvaluator
# from rage.utils.dataset import Dataset


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--env", type=str, default="sokoban", help="Environment name (default: 'sokoban').")
    parser.add_argument("--num_traj", type=int, default=500, help="Number of trajectories to generate (default: 100).")
    parser.add_argument("--algo", type=str, default="bfs", choices=["bfs"], help="Algorithm to use (default: 'bfs').")
    parser.add_argument("--seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/train-trajectories.json", help="Output file to save the trajectories (default: 'data/train-trajectories.json').")
    parser.add_argument("--train_size", type=int, default=3000, help="Number of trajectories to generate (default: 3000).")
    parser.add_argument("--test_size", type=int, default=100, help="Number of trajectories to generate (default: 100).")
    parser.add_argument("--bfs_max_nodes", type=int, default=1000, help="Maximum number of nodes to use for BFS (default: 100000).")
    args = parser.parse_args()
    
    assert args.env == "sokoban", "Unsupported environment: {args.env}"
    assert args.algo == "bfs", "Unsupported algorithm: {args.algo}"
    data_source = args.env

    # env = SokobanEnv(dim_room=(6, 6), num_boxes=2, max_steps=10)
    # policy = BFSPolicy(max_nodes=args.bfs_max_nodes)
    # evaluator = TrajectoryEvaluator(env, policy, max_steps=10)

    # Generate trajectories
    seeds = range(args.seed, args.seed + args.num_traj)
    # trajectories = evaluator.batch_evaluate(seeds, mp=False)

    # # Print metrics for BFS
    # evaluator.print_metrics()

    # TRAIN_SIZE = args.train_size
    # TEST_SIZE = args.test_size

    # dataset_views = Dataset(trajectories).transform("decision_making")

    # assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    # train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    # test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    # dataset just need to provide placeholders
    def _create_instance(idx):
        """
        Actually, we are not using any information from the trajectories now except the random seed index, but just launch the envs in the training rollout process.
        """
        return {
            "data_source": data_source,
            "prompt": [{"role": "system", "content": "You are a helpful assistant."}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": idx}
        }
    train_dataset = Dataset.from_list([_create_instance(i) for i in range(args.train_size)])
    test_dataset = Dataset.from_list([_create_instance(i) for i in range(args.test_size)])


    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn

    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))

if __name__ == "__main__":
    main()