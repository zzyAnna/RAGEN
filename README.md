# Agent-R1

Agent-R1 is a reproduction of the DeepSeek-R1(-Zero) methods for *training agentic models*.




## Setup
1. setup with (private) scripts from https://github.com/ZihanWang314/setup-new-env/blob/main/initialize.sh, L1-L40;
2. init environment:
```bash
conda create -n ragen python=3.9 -y
conda activate ragen

git clone git@github.com:ZihanWang314/agent-r1.git
cd agent-r1

# setup install
pip install -e . # includes verl-rage-ext (by us) and verl-core (by the verl team)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt # other packages

# flash attention 2
pip3 install flash-attn --no-build-isolation
# if flash attn fails, you may need to install cuda-toolkit first
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX # /opt/conda/envs/zero
pip3 install flash-attn --no-build-isolation
```


## basic processes

Create data:
```bash
# sokoban env settings. will determine game difficulty
# it's normal to see some SOKOBAN errors, but the data will be created and it's fine

# below is buggy

export DIM_X=6
export DIM_Y=6
export NUM_BOXES=1
export MAX_STEPS=5
export SEARCH_DEPTH=30
# python scripts/dataset_curation.py \
#     --output data/sokoban

# 
```

Export variables:
```bash
export DATA_DIR=data/sokoban
export BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export EXPERIMENT_NAME=test-qwen2.5-0.5b-instruct-1mbsz
export MICRO_BATCH_SIZE=1
export TRAIN_BATCH_SIZE=128 # 256
export PPO_BATCH_SIZE=64 # 128
export MAX_START_LENGTH=400 # the first round prompt max length
export MAX_RESPONSE_LENGTH=100
export MAX_OBS_LENGTH=120
export MAX_TURNS=5
export NUM_UPDATE_PER_ROLL=1 # roll out for a batch, then the model do N times of update. Currently not implemented.
export LOG_MODE="['wandb']" # or 'console'
export GCP=True # gradient checkpointing
bash ./train.sh # more arguments in this file

# default config file is verl/trainer/config/ppo_trainer.yaml

```


Or for 3B:
```bash
export DATA_DIR=data/sokoban
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export EXPERIMENT_NAME=test-qwen2.5-3b-instruct-1mbsz
export MICRO_BATCH_SIZE=1
export MAX_START_LENGTH=400 # the first round prompt max length
export MAX_RESPONSE_LENGTH=400
export MAX_OBS_LENGTH=200
export MAX_TURNS=5
export NUM_UPDATE_PER_ROLL=1 # roll out for a batch, then the model do N times of update. Currently not implemented.
export LOG_MODE="['wandb']" # or 'console'
# default config file is verl/trainer/config/ppo_trainer.yaml

bash ./train.sh # more arguments in this file
```
