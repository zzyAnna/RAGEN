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
pip install -e . # includes verl-ragen (by us) and verl-core (by the verl team)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121


# Optional: to install flash-attn, you may need to install cuda-toolkit first if you don't have
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX # /opt/conda/envs/zero
pip3 install flash-attn --no-build-isolation



pip install -r requirements.txt # other packages

```


## basic processes

Create data:
```bash
# sokoban env settings. will determine game difficulty
# it's normal to see some SOKOBAN errors, but the data will be created and it's fine


export DIM_X=6
export DIM_Y=6
export NUM_BOXES=1
export MAX_STEPS=5
export SEARCH_DEPTH=30


python scripts/dataset_curation.py \
    --output data/sokoban \
    --seed 10000 \
    --train_size 10000 \
    --test_size 10 \
    --prefix qwen-instruct

# 
```

Export variables:
```bash
export DATA_DIR=data/sokoban
export DIM_X=6
export DIM_Y=6
export NUM_BOXES=1
export MAX_STEPS=5
export SEARCH_DEPTH=30

# export CUDA_VISIBLE_DEVICES=0
# export BASE_MODEL=Qwen/Qwen2.5-0.5B
# export EXPERIMENT_NAME=test-qwen2.5-0.5b

# export CUDA_VISIBLE_DEVICES=1
# export BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
# export EXPERIMENT_NAME=test-qwen2.5-0.5b-instruct

export CUDA_VISIBLE_DEVICES=1
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export EXPERIMENT_NAME=test-r1-distill-1.5b

export CUDA_VISIBLE_DEVICES=0,1
export BASE_MODEL=Qwen/Qwen2.5-3B
export EXPERIMENT_NAME=test-qwen2.5-3b

# TODO: Run this
# export CUDA_VISIBLE_DEVICES=1
# export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
# export EXPERIMENT_NAME=test-qwen2.5-3b-Instruct


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
export N_GPUS=1
export ROLLOUT_TP_SIZE=1

bash ./train.sh # more arguments in this file

# default config file is verl/trainer/config/ppo_trainer.yaml

```


## Visualization
1. By setting arguments in `train.sh`, you can visualize the trajectory:
```bash
logging.log_images=True # set to True to log images
logging.log_image_dir=.log.debug/trajectory # set to the directory to save images
logging.log_image_step_size=1 # save image every _ steps
logging.log_n_image_per_batch=8 # save _ images per batch   
```

2. Example image for one trajectory: 
<p align="center" style="display: flex; justify-content: center; gap: 10px;">
    <img src="./public/step_1.png" width="200px" alt="s" />
    <img src="./public/step_2.png" width="200px" alt="s" />
    <img src="./public/step_3.png" width="200px" alt="s" />
    <img src="./public/step_4.png" width="200px" alt="s" />
    <img src="./public/step_5.png" width="200px" alt="s" />
</p>






# TODO: Cases



# Why we give (s1 | a1 s2 a2 s3 a3) as input?
1. 区分rollout和train: rollout的时候是给s生成a, 多次循环；train的时候是给s1生成后面的
几个好处：1 多轮统一，不会搞出新的instance，让batchsize不稳定
2. 能多学一点儿state，可能能做planning










