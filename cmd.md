# Base Experiments

## Bandit
### `RAGEN` *Qwen2.5-0.5B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh two_armed_bandit \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=two_armed_bandit_0_5B_instruct_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=32 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=1 \
    training.n_rollout=1 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/two_armed_bandit_0_5B_instruct_ragen_main.log
```
### `RAGEN` *Qwen2.5-3B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh two_armed_bandit \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=two_armed_bandit_3B_instruct_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=32 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=1 \
    training.n_rollout=1 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/two_armed_bandit_3B_instruct_ragen_main.log
```

### `RAGEN w/o thinking` *Qwen2.5-0.5B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh two_armed_bandit \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=two_armed_bandit_ragen_no_think_rl \
    training.micro_batch_size=4 \
    training.train_batch_size=32 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=1 \
    training.n_rollout=1 \
    training.total_training_steps=200 \
    training.no_think_rl=True \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/two_armed_bandit_0_5B_instruct_ragen_no_think.log
```
### `RAGEN w/o thinking` *Qwen2.5-3B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh two_armed_bandit \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=two_armed_bandit_ragen_no_think_rl \
    training.micro_batch_size=4 \
    training.train_batch_size=32 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=1 \
    training.n_rollout=1 \
    training.total_training_steps=200 \
    training.no_think_rl=True \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/two_armed_bandit_3B_instruct_ragen_no_think.log
```

## Sokoban

### `RAGEN` *Qwen2.5-0.5B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=sokoban_0_5B_instruct_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_0_5B_instruct_ragen_main.log
```

### `RAGEN` *Qwen2.5-3B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=sokoban_3B_instruct_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_3B_instruct_ragen_main.log
```

### `RAGEN w/o thinking` *Qwen2.5-0.5B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=sokoban_0_5B_instruct_ragen_no_think \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.no_think_rl=True \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_0_5B_instruct_ragen_no_think.log
```

### `RAGEN w/o thinking` *Qwen2.5-3B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=sokoban_3B_instruct_ragen_no_think \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.no_think_rl=True \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_3B_instruct_ragen_no_think.log
```

### `SFT` *Qwen2.5-0.5B-Instruct*
**TODO** sft config needs to be added to train.py
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    rl_or_sft=sft \
    sft.training.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    sft.training.experiment_name=sokoban_0_5B_instruct_sft \
    sft.data_generation.train_size=1000 \
    sft.data_generation.test_size=100 \
    sft.training.micro_batch_size=4 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_0_5B_instruct_sft.log
```

### `SFT` *Qwen2.5-3B-Instruct*
**TODO** sft config needs to be added to train.py
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    rl_or_sft=sft \
    sft.training.base_model=Qwen/Qwen2.5-3B-Instruct \
    sft.training.experiment_name=sokoban_3B_instruct_sft \
    sft.data_generation.train_size=1000 \
    sft.data_generation.test_size=100 \
    sft.training.micro_batch_size=4 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_3B_instruct_sft_1.log
```

## FrozenLake

### `RAGEN` *Qwen2.5-0.5B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=frozenlake_0_5B_instruct_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_0_5B_instruct_ragen_main.log
```
### `RAGEN` *Qwen2.5-3B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=frozenlake_3B_instruct_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_3B_instruct_ragen_main.log
```

### `RAGEN w/o thinking` *Qwen2.5-0.5B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=frozenlake_0_5B_instruct_ragen_no_think \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.no_think_rl=True \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_0_5B_instruct_ragen_no_think.log
```

### `RAGEN w/o thinking` *Qwen2.5-3B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=frozenlake_3B_instruct_ragen_no_think \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.no_think_rl=True \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_3B_instruct_ragen_no_think.log
```

### `SFT` *Qwen2.5-0.5B-Instruct*
**TODO** sft config needs to be added to train.py
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    rl_or_sft=sft \
    sft.training.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    sft.training.experiment_name=frozenlake_0_5B_instruct_sft \
    sft.data_generation.train_size=1000 \
    sft.data_generation.test_size=100 \
    sft.training.micro_batch_size=4 \
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_0_5B_instruct_sft.log
```

### `SFT` *Qwen2.5-3B-Instruct*
**TODO** sft config needs to be added to train.py
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    rl_or_sft=sft \
    sft.training.base_model=Qwen/Qwen2.5-3B-Instruct \
    sft.training.experiment_name=frozenlake_3B_instruct_sft \
    sft.data_generation.train_size=1000 \
    sft.data_generation.test_size=100 \
    sft.training.micro_batch_size=4 \
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_3B_instruct_sft.log
```


## Analysis

### Model Scaling - `Sokoban`

#### `RAGEN` *Qwen2.5-0.5B-Instruct*
Refer to section [Sokoban 0.5B Instruct RAGEN](#sokoban)

#### `RAGEN` *Qwen2.5-1.5B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-1.5B-Instruct \
    model.experiment_name=sokoban_1_5B_instruct_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_1_5B_instruct_ragen_main.log
```

#### `RAGEN` *Qwen2.5-3B-Instruct*
Refer to section [Sokoban 3B Instruct RAGEN](#sokoban)

#### `RAGEN` *Qwen2.5-7B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    system.n_gpus=2 \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=sokoban_7B_instruct_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_7B_instruct_ragen_main.log
```

### Model Scaling - `Sokoban`
**TODO** inference and model checkpoint loading ... 25 turns at most
```bash
mkdir -p ./log/terminal

```

### 