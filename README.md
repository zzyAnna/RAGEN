# agent-r1
1. setup with (private) scripts from https://github.com/ZihanWang314/setup-new-env/blob/main/initialize.sh, L1-L40;
2. init environment:
```bash
conda create -n zero python=3.9 -y
conda activate zero

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX # /opt/conda/envs/zero

# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray
pip3 intall peft
# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib


# setup TinyZero
git clone git@github.com:Jiayi-Pan/TinyZero.git
cd TinyZero
pip install -e .
```


## basic processes

Create data:
```bash
cd TinyZero
python ./examples/data_preprocess/countdown.py --local_dir countdown_data
```

Export variables:
```bash
export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B
export DATA_DIR=countdown_data
export ROLLOUT_TP_SIZE=2
export VLLM_ATTENTION_BACKEND=XFORMERS
export EXPERIMENT_NAME=countdown-qwen2.5-3b-grad_ckpt

bash ./scripts/train_tiny_zero.sh
```