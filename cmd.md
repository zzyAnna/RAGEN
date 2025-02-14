## Base Experiments

### Bandit
`RAGEN`
```bash
mkdir -p ./log/terminal

bash train.sh two_armed_bandit \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=two_armed_bandit_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=32 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=1 \
    training.n_rollout=1 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/two_armed_bandit_ragen_main.log
```

`RAGEN w/o thinking`
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
    optimization.adv_estimator=brpo > ./log/terminal/two_armed_bandit_ragen_no_think.log
```
### Sokoban
`RAGEN`
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=sokoban_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_ragen_main.log
```
`RAGEN w/o thinking`
```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=sokoban_ragen_no_think_rl \
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
    optimization.adv_estimator=brpo > ./log/terminal/sokoban_ragen_no_think.log
```

### FrozenLake
`RAGEN`
```bash
mkdir -p ./log/terminal
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=frozenlake_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=4 \
    training.ppo_batch_size=32 \
    training.val_data_num=50 \
    training.max_turns=5 \
    training.n_rollout=8 \
    training.total_training_steps=200 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_ragen_main.log
```
`RAGEN w/o thinking`
```bash
mkdir -p ./log/terminal

bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=frozenlake_ragen_no_think_rl \
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
    optimization.adv_estimator=brpo > ./log/terminal/frozenlake_ragen_no_think.log
```