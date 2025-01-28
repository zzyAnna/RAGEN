# RAGEN

<p align="center" style="font-size: 18px;">
  <strong>RAGEN</strong> is the first reproduction of the <strong>DeepSeek-R1(-Zero)</strong> methods for <em>training agentic models</em>.<br>
  We strongly believe in the future of RL + LLM + Agents. The release is a minimally viable leap forward.
</p>


## Framework

<img src="./public/framework.png" width="800px" alt="s" />



## Performance

We evaluate RAGEN on Qwen-2.5-{0.5B, 3B}-{Instruct, None} and DeepSeek-R1-Distill-Qwen-1.5B, on the [Gym-Sokoban](https://github.com/mpSchrader/gym-sokoban?tab=readme-ov-file) task. The maximum reward of this environment is **10.9**.

<img src="./public/loss_curve.png" width="800px" alt="s" />

The loss curve have not converged (since our compute is currently limited...). But we already see some trends:
 - Instruct-finetuned models are not significantly advantaged ahead Pretrained-only models, although they are better at start.
 - 3B models are performing better than 0.5B models as well, but the advantages are also not that obvious at around 40 steps.
 - Interestingly, R1-distilled 1.5B model do less well than 0.5B models for now.

We prepare to release a complete wandb plot for these experiment runs, although you can try it your own and it may even be faster than our run (reasons ahead).









## Environment Setup

```bash
conda create -n ragen python=3.9 -y
conda activate ragen

git clone git@github.com:ZihanWang314/ragen.git
cd ragen

# setup install
pip install -e . # includes verl-ragen (by us) and verl-core (by the verl team)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121


# Optional: to install flash-attn, you may need to install cuda-toolkit first if you don't have
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX # /opt/conda/envs/zero
pip3 install flash-attn --no-build-isolation



pip install -r requirements.txt # other packages

```


## Train Models

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
    --prefix qwen-instruct # we find it works well for base models
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

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL=checkpoints/Agent-R1/test-qwen2.5-0.5b-instruct-1mbsz/actor/global_step_100
export EXPERIMENT_NAME=test-qwen2.5-0.5b-imagetest


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






## Cases
Please see cases/ file.
There are only limited cases for now, including [reward hacking](https://github.com/ZihanWang314/agent-r1/blob/main/cases/reward_hacking.txt) and the [suck moment](https://github.com/ZihanWang314/agent-r1/blob/main/cases/suck_moment.txt). we will add more cases recently.


## Authors

[Zihan Wang*](https://zihanwang314.github.io/)

[Kangrui Wang](https://jameskrw.github.io/)

[Qineng Wang](https://qinengwang-aiden.github.io/)

[Pingyue Zhang](https://williamzhangsjtu.github.io/)

[Manling Li†](https://limanling.github.io)

*: Project Lead; †: Advising.
Remaining authors are alphabetical order.



## Acknowledgements


We thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) for providing the DeepSeek-R1 model and ideas. We thank the [veRL](https://github.com/volcengine/verl) team for their infrastructure. We thank the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) team for their discoveries that inspired our early exploration. We thank Yiping Lu, Runxin Xu, Kyunghyun Cho for insightful discussions with them.
