# A comprehensive guide on ReLAX 
PS. ReLAX is the original name of RAGE.

## Quick Start

### Installation

```bash
cd relax-agent
pip install -e .
```


### Usage

You can directly try the following code:
```python
from relax.policy import BFSPolicy, FixedPolicy, HeuristicPolicy, LLMPolicy
from relax.env.sokoban.sokoban import SokobanEnv, ACTION_LOOKUP
from relax.evaluators.trajectory_evaluator import TrajectoryEvaluator

env = SokobanEnv(dim_room=(6, 6), max_steps=10, num_boxes=2, search_depth=10)
policy = LLMPolicy(model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", temperature=0.7)
evaluator = TrajectoryEvaluator(env, policy, max_steps=10)

trajectories = evaluator.batch_evaluate(seeds=range(200), mp=False)
evaluator.print_metrics()

for i in trajectories[0]: print(i['observation'] + '\n' + ACTION_LOOKUP[i['action']])




# env, policy = SokobanEnv(), BFSPolicy()
# trajectories = evaluator.batch_evaluate(seeds=range(200), mp=True) 

# env, policy = SokobanEnv(), LLMPolicy(model_path="Qwen/Qwen2.5-3B-Instruct")
```

On the other hand, you can also try:
```bash
python main.py
```

## Codebase Walkthrough

This repo mainly shows how to incorporate ReLAX into a existing agent.
You will have an agent trained on different capabilities, and you can use ReLAX to make it perform better.

### Overview

> First stage:
> Train a model with different capabilities (decision_making, forward dynamics, etc). 
> Evaluate such model with baselines, e.g. MCTS-Agent and Action-Agent. <br>
> Second stage:
> Use ReLAX to incorporate these capabilities into the agent.


### Stage 1: Capability Training and Baseline Models



#### Data synthesis
First, curate a ground-truth dataset using BFS.
```bash

---This below script is currently buggy---
python scripts/dataset_curation.py \
    --env sokoban --num_traj 3000 --seed 10000 \
    --output data/sokoban_train.json
# Success rate: 100.0%, Average steps: 10.66

python scripts/dataset_curation.py \
    --env sokoban --num_traj 1000 --seed 10000 \
    --output data/sokoban_train_1000.json

python scripts/dataset_curation.py \
    --env sokoban --num_traj 200 --seed 0 \
    --output data/sokoban_eval.json
# Success rate: 100.0%, Average steps: 10.61

python scripts/dataset_curation.py \
    --env sokoban --num_traj 32 --seed 0 \
    --output data/sokoban_minimaltest_10.json
# Success rate: 100.0%, Average steps: 8.90
```

<details>
<summary><h5 style="display: inline;">Dataset usage and schema</p></summary>

How to use `Dataset` viewer to get different views of the dataset (e.g., forward-dynamics, transition-modeling, ...):

```python
from relax.utils.dataset import Dataset
import json
with open("data/sokoban_eval.json", "r") as f: 
      raw_data = json.load(f)

dataset = Dataset(raw_data, max_examples_per_traj=3) # For each trajectory, they are transformed to single-step training examples, and each trajectory appear at most 3 examples.
tp_dataset = dataset.transform("task_planning")
```

Dataset instance schema:
```python
{
    'step': int,  # Current step number in episode
    'observation': str,  # Current state as a tab-separated grid string
    'observation-list': List[str],  # Current state as list of rows
        # Each string represents one row of the maze
        # Symbols: #=Wall, *=Empty, O=Goal, X=Box, P=Player, √=Box on Goal
    'action': int,  # Action taken (1=up, 2=down, 3=left, 4=right)
    'reward': float,  # Immediate reward for action
    'next_observation': str,  # State after action (same format as 'observation')
    'done': bool,  # Whether episode is complete
    'success': bool,  # Whether goal was achieved
    'all-observation': str,  # History of states (same format as 'observation')
    'all-observation-list': List[List[str]],  # History of states as nested lists
        # Outer list: timesteps
        # Inner list: rows of maze at each timestep
    'best_future_trajectory': List[Tuple[str, int]],
        # List of (state, action) pairs showing optimal future path
        # state: maze state as string (same format as 'observation')
        # action: next action to take from that state
}

# Maze Symbol Legend:
# # = Wall
# O = Goal
# X = Box
# P = Player
# √ = Box on Goal (completed goal)
# _ = Empty space
# S = Player on Goal

# Action Legend:
# 1 = Up
# 2 = Down
# 3 = Left
# 4 = Right
```

```python
# tp_dataset instance schema
{
    'strategy': str,  # Type of planning strategy
    'item_id': str,  # Unique identifier (can be empty)
    'conversations': List[Dict],  # List of conversation turns, in openai-compatible format
    'prompt': str,  # For non-chat models, it is the same as the input
    'prediction': str,  # For non-chat models, space-separated sequence of numbers representing actions
    'metadata': Dict,
        {
            'prompt_length': int,    # Length of the prompt
            'prediction_length': int  # Length of the prediction
        }
}
```
</details>

#### Baselines

We have provided two baselines:

> Action-Agent: A simple agent that trained on action sequencing and use direct action prediction.

```bash
# DEBUG
python scripts/train_on_views.py \
    --dataset_file data/sokoban_train.json \
    --lora --views decision_making \
    --config ./config/qwen2_5_3B_lora_relax.yaml \
    output_dir=.cache/qwen2_5_3B_action_agent_debug_notraininputs/lora_single_device

python scripts/evaluate_on_views.py \
    --dataset_path data/sokoban_eval.json \
    --model_path .cache/qwen2_5_3B_action_agent_debug_notraininputs/lora_single_device/epoch_0 \
    --views decision_making \
    --max_new_tokens 10 \
    --save_predictions \
    --output_dir evaluation_results

python scripts/evaluate_on_envs.py \
    --seed_range 10000 10200 \
    --model_path .cache/qwen2_5_3B_action_agent_debug_notraininputs/lora_single_device/epoch_0 \
    --save_predictions \
    --output_dir evaluation_results/qwen2_5_3B_action_agent_debug_notraininputs
    # --seed_range 0 200 \ # basically the eval set

    
```


```bash
# Simple version
python scripts/train_on_views.py \
    --dataset_file data/sokoban_train.json \
    --lora --views decision_making \
    --config ./config/qwen2_5_3B_lora_relax.yaml \
    output_dir=.cache/qwen2_5_3B_action_agent/lora_single_device

python scripts/evaluate_on_views.py \
    --dataset_path data/sokoban_eval.json \
    --model_path .cache/qwen2_5_3B_action_agent/lora_single_device/epoch_0 \
    --views decision_making \
    --max_new_tokens 10 \
    --save_predictions \
    --output_dir evaluation_results/qwen2_5_3B_action_agent/views

python scripts/evaluate_on_envs.py \
    --seed_range 0 200 \
    --model_path .cache/qwen2_5_3B_action_agent/lora_single_device/epoch_0 \
    --save_predictions \
    --output_dir evaluation_results/qwen2_5_3B_action_agent/env

```

<details>
<summary><h5 style="display: inline;">Detailed version</p></summary>

```bash
# training an action agent
python scripts/train_on_views.py \
    --dataset_file data/sokoban_train.json \
    --views decision_making \
    --lora \
    --config ./config/qwen2_5_3B_lora_relax.yaml \
    tokenizer.max_seq_len=4096 \
    gradient_accumulation_steps=8 \
    epochs=1 \
    batch_size=2 \
    output_dir=.cache/qwen2_5_3B_action_agent/lora_single_device
    
python scripts/evaluate_on_views.py \
    --dataset_path data/sokoban_eval.json \
    --model_path .cache/qwen2_5_3B_action_agent/lora_single_device/epoch_0 \
    --views decision_making \
    --batch_size -1 \
    --tensor_parallel_size 1 \
    --temperature 0.0 \
    --max_new_tokens 256 \
    --save_predictions \
    --output_dir evaluation_results


```
</details>


> MCTS-Agent: A MCTS-based agent that trained on world modeling and action sequencing, and use MCTS to select actions.

```bash
# Simple version
python scripts/train_on_views.py \
    --dataset_file data/sokoban_train.json \
    --lora --views forward_dynamics \
    --config ./config/qwen2_5_3B_lora_relax.yaml \
    output_dir=.cache/qwen2_5_3B_mcts_agent/lora_single_device



python scripts/evaluate_on_views.py \
    --dataset_path data/sokoban_eval.json \
    --model_path .cache/qwen2_5_3B_mcts_agent/lora_single_device/epoch_0 \
    --views forward_dynamics \
    --save_predictions \
    --output_dir evaluation_results/qwen2_5_3B_mcts_agent/views

python scripts/evaluate_on_envs.py \
    --seed_range 0 200 \
    --model_path .cache/qwen2_5_3B_mcts_agent/lora_single_device/epoch_0 \
    --save_predictions \
    --output_dir evaluation_results/qwen2_5_3B_mcts_agent/env

```

<details>
<summary><h5 style="display: inline;">Detailed version</p></summary>

```bash
# training an action agent
python scripts/train_on_views.py \
    --dataset_file data/sokoban_train.json \
    --views decision_making forward_dynamics \
    --lora \
    --config ./config/qwen2_5_3B_lora_relax.yaml \
    tokenizer.max_seq_len=4096 \
    gradient_accumulation_steps=8 \
    epochs=1 \
    batch_size=2 \
    output_dir=.cache/qwen2_5_3B_mcts_agent/lora_single_device

python scripts/evaluate_on_views.py \
    --dataset_path data/sokoban_eval.json \
    --model_path .cache/qwen2_5_3B_mcts_agent/lora_single_device/epoch_0 \
    --views task_planning decision_making forward_dynamics\
    --batch_size 32 \
    --tensor_parallel_size 1 \
    --temperature 0.0 \
    --max_new_tokens 256 \
    --save_predictions \
    --output_dir evaluation_results
```
</details>

> other baselines: distilling from DeepSeek-R1 or RL with DeepSeek-R1

# Analytical Exploration (Below is under development)

In below script, you will see how different intermediate states exposed to the model can help the model to make better predictions.
```bash
python scripts/eval_rollouts.py 
```


Now, you can evaluate current large models' ability on these views, or train a small model on the dataset views:
```bash
bash scripts/train_on_views.sh
bash scripts/evauate_on_views.sh
```
```md
>>> Output
2025-01-06 20:14:23,782 - ERROR - Model checkpoint of size 3.70 GiB saved to /tmp/torchtune/qwen2_5_3B/lora_single_device/epoch_0/ft-model-00001-of-00002.safetensors
2025-01-06 20:14:23,782 - ERROR - 1|57|Loss: 0.061056770384311676: 100%|██████████| 57/57 [20:46<00:00, 21.88s/it]
```






Relax is a new paradigm that try to enable agent the ability to structured reasoning and planning. While MCTS focus on running out trajectories and evaluate how well are these trajectories, Relax does a more forward-looking approach by estimating the possible checkpoints in the future. For example, at state_t, MCTS would sample 10 trajectories to state_t+k, and evaluate these trajectories. However, Relax would give 10 different estimations as "patches", each creating new assertions or amending current assertions of what will happen in the future. For example, at state_t, the agent could begin by estimating state_t+1, state_t+2, then state_last, anything that may be possibly true. After a certain points or amendments, they start to estimate a_t. 

# Analytical exploratory experiments: which trajectory helps?
In this section, I'll try to first let you see current dataset format. Then, I'll write a script analyzing how would current models do well based on different estimations. Note that there is a bug that actual performance is "the ability to make good predictions" x "the ability to predict at the best place". We will not directly quantize the first term, just assume the ability to make good prediction is very good, and focus on the second term.

Since we do not have direct estimation on rewards, we just focus on predicting three possible later stages, making it a list, each time inserting something into the list or after it, and check about the loss to make good predictions on ground-truth action at this time.


## Evaluate different policies in scale
```bash
# this needs to be revised later since it is not good for adapting different policies.
python -m relax.policy.classic.rule_based # run best policy test
python -m relax.evaluate # run evaluation test on 1000 worlds
```


## Learn more on the Sokoban environment
```bash
# see sokoban intro. Will print to the command terminal
python -m relax.env.sokoban.sokoban 

# run the sokoban demo on http://localhost:5000
python -m relax.env.sokoban.demo
```

python relax/evaluate.py  --policy relax-agent   ++model_name  relax-agent/relax-v0.1-llama3.1-instruct-1b

###################
you can try using relaxation agent to solve problems by using following evaluation script.
python relax/evaluate.py  --policy relax-agent   ++model_name  relax-agent/relax-v0.1-llama3.1-instruct-1b

you can also try with baselines:
python relax/evaluate.py  --policy mcts-agent   ++model_name  relax-agent/mcts-llama3.1-instruct-1b    ++evaluator_name  relax-agent/mcts-evaluator-llama3.1-instruct-1b

or with:
python relax/evaluate.py  --policy cot-agent   ++model_name  openai/gpt-4o-mini-2024-07-18

the agents are trained on the sokoban environment. 





###################
In the next section, we show an important breakthrough in our research: How to train a relax-agent?

The agent need to learn how to accurately estimate relaxed goals (subgoals, states, actions, etc) and also ...

you just need to provide the SFT data (you can download it [here] or compile it through 
```bash
python -m ascal.env.compile.relaxation \
      --env sokoban   --seed 50000   --n_data 1000   \
      --save_dir .cache/sokoban_relaxation
```

if you want to train on your own dataset jsonl file like: 
{"id": "[ID]", "trajectory": [{"state": "[STATE]", "action": "[ACTION]", "reward": "None" or "[REWARD]"}, ...]}
and you have a prompt template like:
""
you can also consider rendering from template and trajectory
```bash
python -m ascal.env.compile.relaxation \
      --file_path FILE_PATH  --prompt_template PROMPT_PATH   \
      --save_dir .cache/dataset_relaxation
```



