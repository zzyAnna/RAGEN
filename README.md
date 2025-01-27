# agent-r1
1. setup with (private) scripts from https://github.com/ZihanWang314/setup-new-env/blob/main/initialize.sh, L1-L40;
2. init environment:
```bash
conda create -n agent python=3.9 -y
conda activate agent


# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray
pip3 install peft
# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib


# if flash attn fails, you may need to install cuda-toolkit first
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX # /opt/conda/envs/zero

git clone git@github.com:ZihanWang314/agent-r1.git
cd agent-r1
# setup install
pip install verl[full] # includes verl-core (by the verl team) and verl-rage-ext (by us)
```


## basic processes

Create data:
```bash
# it's normal to see some SOKOBAN errors, but the data will be created
python scripts/dataset_curation.py \
    --output data/sokoban
```

Export variables:
```bash
bash ./train.sh
```


Or for 3B:
```bash
export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B
export DATA_DIR=countdown_data
export ROLLOUT_TP_SIZE=2
export VLLM_ATTENTION_BACKEND=XFORMERS
export EXPERIMENT_NAME=countdown-qwen2.5-3b-grad_ckpt
export MICRO_BATCH_SIZE=8
export RESPONSE_LENGTH=1024
export LOG_MODE=wandb
export MULTI_PROCESSING=ray
```
