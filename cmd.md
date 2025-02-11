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
    model.experiment_name=test_zihan_brpo_p8r16m32 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=32 \
    training.micro_batch_size=2 \
    optimization.adv_estimator=brpo \
    training.use_kl_loss=True
    # train_batch_size: rollout prompts
    # n_rollout: responses for each prompt
    # ppo_batch_size: update things
    # consider making the "epoch X step X" as "Rollout step X, update step X*X"?
    # grpo | brpo | apo
    # effective batch size: training.train_batch_size * training.n_rollout
```

