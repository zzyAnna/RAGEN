```bash
bash train.sh sokoban \
    model.experiment_name=new_test

# override config
bash train.sh sokoban \
    model.experiment_name=test_zihan \
    training.train_batch_size=8 \
    training.ppo_batch_size=4

# For developers, if you want to add your own config keys, please check [ base.yaml | train.sh | ragen/train.py | verl/trainer/config/ppo_trainer.yaml | and the main_ppo.py in verl/trainer/ppo ] to make sure the changes are reflected coherently.
```

Below:Base experiment -> figure X in paper, aiming to xxx

```bash
bash train.sh sokoban \
    model.experiment_name=XXXX \
    argument ...
```


Below:Base experiment

```bash
bash train.sh sokoban \
    model.experiment_name=test_base
```

Below:GRPO

```bash
bash train.sh sokoban \
    model.experiment_name=test_zihan_grpo \
    training.n_rollout=64 \
    training.train_batch_size=16 \
    training.ppo_batch_size=16 \
    optimization.advantage_estimator=grpo
    # effective batch size: training.train_batch_size * training.n_rollout
```

