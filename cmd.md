# Base Experiments

## Hyperparameter search.
We first do hyperparameter search, hoping to find a good combination to guide later experiment settings.

**[Note]** Current multi-GPUs strategy is **FSDP**. We are running with **3B** models.

#### [BUDGET]: In total: ***27*** runs, Qwen2.5-3B-Instruct, Sokoban
- Search group 1: 9 runs
    - train_batch_size: [4, 8, 16]
    - n_rollout: [4, 8, 16]
- Search group 2: 4 runs
    - actor_lr: [5e-7, 1e-6, 5e-6, 1e-5]
- Search group 3: 5 runs
    - kl_coef: [0.001, 0.005, 0.01, 0.04, 0.1, 0.5]
- Search group 4: 9 runs
    - max_turns: [2, 5, 8]
    - temperature: [0.1, 0.5, 1]

## **[WARNING] If `--parallel` mode fails (really because of gpu, not out of memory), please remove this argument**
#### [EXP 1]: Search group 1 [train_batch_size, n_rollout] (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=two_armed_bandit \
    --exp_base_name="hyperparam_searching" \
    --search_group 1 \
    --micro_batch_size 4 \
    --parallel \
    --n_gpus 1
```
#### [EXP 2]: Search group 2 [actor_lr] (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=two_armed_bandit \
    --exp_base_name="hyperparam_searching" \
    --search_group 2 \
    --micro_batch_size 4 \
    --parallel \
    --n_gpus 1
```
#### [EXP 3]: Search group 3 [kl_coef]  (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=two_armed_bandit \
    --exp_base_name="hyperparam_searching" \
    --search_group 3 \
    --micro_batch_size 4 \
    --parallel \
    --n_gpus 1
```
#### [EXP 4]: Search group 4 [max_turns, temperature] (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=two_armed_bandit \
    --exp_base_name="hyperparam_searching" \
    --search_group 4 \
    --micro_batch_size 4 \
    --parallel \
    --n_gpus 1
```

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
    training.max_turns=1 \
    training.n_rollout=1 \
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
    training.max_turns=1 \
    training.n_rollout=1 \
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
    training.max_turns=1 \
    training.n_rollout=1 \
    training.no_think_rl=True \
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
    training.max_turns=1 \
    training.n_rollout=1 \
    training.no_think_rl=True \
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
    training.max_turns=5 \
    training.n_rollout=8 \
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
    training.max_turns=5 \
    training.n_rollout=8 \
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
    training.max_turns=5 \
    training.n_rollout=8 \
    training.no_think_rl=True \
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
    training.max_turns=5 \
    training.n_rollout=8 \
    training.no_think_rl=True \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_3B_instruct_ragen_no_think.log
```

### `SFT` *Qwen2.5-0.5B-Instruct*
- `training.val_batch_size`: batch size for validation after sft
- `training.val_data_num`: number of validation data after sft, None means all
- `training.n_rollout`: number of rollout agents when validating after sft
**NOTE** Validate on validation set for RL
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    rl_or_sft=sft \
    sft.output_dir=models/sft/sokoban/Qwen2.5-0.5B-Instruct \
    sft.training.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    sft.training.experiment_name=sokoban_0_5B_instruct_sft \
    sft.data_generation.train_size=2000 \
    sft.data_generation.test_size=200 \
    sft.training.micro_batch_size=4 \
    sft.training.epochs=5 \
    training.val_batch_size=10 \
    training.val_data_num= \
    training.n_rollout=1 \
    optimization.adv_estimator=brpo 2>&1 | tee ./log/terminal/sokoban_0_5B_instruct_sft.log
```

### `SFT` *Qwen2.5-3B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    rl_or_sft=sft \
    sft.output_dir=models/sft/sokoban/Qwen2.5-3B-Instruct \
    sft.training.base_model=Qwen/Qwen2.5-3B-Instruct \
    sft.training.experiment_name=sokoban_3B_instruct_sft \
    sft.data_generation.train_size=2000 \
    sft.data_generation.test_size=200 \
    sft.training.micro_batch_size=4 \
    sft.training.epochs=5 \
    training.val_batch_size=10 \
    training.val_data_num= \
    training.n_rollout=1 \
    optimization.adv_estimator=brpo 2>&1 | tee ./log/terminal/sokoban_3B_instruct_sft.log
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
    training.max_turns=5 \
    training.n_rollout=8 \
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
    training.max_turns=5 \
    training.n_rollout=8 \
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
    training.max_turns=5 \
    training.n_rollout=8 \
    training.no_think_rl=True \
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
    training.max_turns=5 \
    training.n_rollout=8 \
    training.no_think_rl=True \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_3B_instruct_ragen_no_think.log
```

### `SFT` *Qwen2.5-0.5B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    rl_or_sft=sft \
    sft.output_dir=models/sft/frozenlake/Qwen2.5-0.5B-Instruct \
    sft.training.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    sft.training.experiment_name=frozenlake_0_5B_instruct_sft \
    sft.data_generation.train_size=2000 \
    sft.data_generation.test_size=200 \
    sft.training.micro_batch_size=4 \
    sft.training.epochs=5 \
    training.val_batch_size=10 \
    training.val_data_num= \
    training.n_rollout=1 \
    optimization.adv_estimator=brpo 2>&1 | tee ./log/terminal/frozenlake_0_5B_instruct_sft.log
```

### `SFT` *Qwen2.5-3B-Instruct*
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    rl_or_sft=sft \
    sft.output_dir=models/sft/frozenlake/Qwen2.5-3B-Instruct \
    sft.training.base_model=Qwen/Qwen2.5-3B-Instruct \
    sft.training.experiment_name=frozenlake_3B_instruct_sft \
    sft.data_generation.train_size=2000 \
    sft.data_generation.test_size=200 \
    sft.training.micro_batch_size=4 \
    sft.training.epochs=5 \
    training.val_batch_size=10 \
    training.val_data_num= \
    training.n_rollout=1 \
    optimization.adv_estimator=brpo 2>&1 | tee ./log/terminal/frozenlake_3B_instruct_sft.log
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
    training.max_turns=5 \
    training.n_rollout=8 \
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
    training.max_turns=5 \
    training.n_rollout=8 \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_7B_instruct_ragen_main.log
```

### Model Scaling - `Sokoban`
**TODO** inference and model checkpoint loading ... 25 turns at most
```bash
mkdir -p ./log/terminal

```

### 