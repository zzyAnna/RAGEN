export N_GPUS=1
export DATA_DIR=data/sokoban
export VLLM_ATTENTION_BACKEND=XFORMERS

export ROLLOUT_TP_SIZE=1
export BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
# export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

export EXPERIMENT_NAME=test-qwen2.5-0.5b-instruct
# export EXPERIMENT_NAME=test-deepseek-r1-qwen-1.5b
export MICRO_BATCH_SIZE=1
export MAX_START_LENGTH=300 # the first round prompt max length
export MAX_RESPONSE_LENGTH=100
export MAX_OBS_LENGTH=100
export MAX_TURNS=3
export MAX_PROMPT_LENGTH=$(($MAX_START_LENGTH + ($MAX_RESPONSE_LENGTH + $MAX_OBS_LENGTH) * ($MAX_TURNS - 1)))
echo "MAX_PROMPT_LENGTH: $MAX_PROMPT_LENGTH"

# export MAX_START_LENGTH=200 # the first round prompt max length
# export MAX_RESPONSE_LENGTH=200
# export MAX_OBS_LENGTH=100
# export MAX_TURNS=11
# export MAX_PROMPT_LENGTH=3200

export LOG_MODE="['console']"
export MULTI_PROCESSING=ray # only choice for now

# default config file is verl/trainer/config/ppo_trainer.yaml

python verl/trainer/main_ppo.py \
multi_processing=$MULTI_PROCESSING \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=1312 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.max_start_length=$MAX_START_LENGTH \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=$LOG_MODE \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=100 \
trainer.project_name=Agent-R1 \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 \
max_turns=$MAX_TURNS
#  2>&1 | tee verl_demo.log


