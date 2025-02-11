## Commands to run
***NOTE! Before you run, please make sure you are under `/RAGEN` dir***
### *EXP SET 1*: Hyperparameter search.
We first do hyperparameter search, hoping to find a good combination to guide later experiment settings.
#### [BUDGET]:
- search 
#### [EXP 1]: Search group 1 [ppo_batch_size] (Tested)
```bash
bash hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 1
```
#### [EXP 2]: Search group 2 [train_batch_size, n_rollout] (Tested)
```bash
bash hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 2 \
    --ppo_batch_size <best searched ppo_batch_size>
```
#### [EXP 3]: Search group 3 [kl_coef]  (Tested)
```bash
bash hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 3 \
    --ppo_batch_size <best searched ppo_batch_size> \
    --train_batch_size <best searched train_batch_size> \
    --n_rollout <best searched n_rollout>
```
#### [EXP 4]: Search group 4 [max_turns, temperature] (Tested)
```bash
bash hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 4 \
    --ppo_batch_size <best searched ppo_batch_size> \
    --train_batch_size <best searched train_batch_size> \
    --n_rollout <best searched n_rollout> \
    --kl_coef <best searched kl_coef>
```
#### [EXP 5]: Search group 5 [actor_lr] (Tested)
```bash
bash hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 5 \
    --ppo_batch_size <best searched ppo_batch_size> \
    --train_batch_size <best searched train_batch_size> \
    --n_rollout <best searched n_rollout> \
    --kl_coef <best searched kl_coef> \
    --max_turns <best searched max_turns> \
    --temperature <best searched temperature>
```

### *EXP SET 2*: Main table bandits, refer to paper Table 2
#### [EXP 6]: RAGEN for bandits
```bash

```
#### [EXP 7]: RAGEN w/o thinking for bandits
```bash

```

#### [EXP 8]: All golden
```bash

```

---
## Below are Deprecated cmd lines.



[EXP 2] Below: Main table experiment(1/12), refer to paper Table 2
- env_name: sokoban
- method: ragen
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=main_result_ragen_sokoban_qwen_2.5_7b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True
```

[EXP 3] Below: Main table experiment(2/12), refer to paper Table 2
- env_name: frozen lake
- method: ragen
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=main_result_ragen_frozenlake_qwen_2.5_7b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True
```

[EXP 4] Below: Main table experiment(3/12), refer to paper Table 2
- env_name: bandit
- method: ragen
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
bash train.sh bandit \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=main_result_ragen_bandit_qwen_2.5_7b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True
```

[EXP 5] Below: Main table experiment(4/12), refer to paper Table 2
- env_name: sokoban
- method: ragen w/o thinking
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
waiting to be done
```

[EXP 6] Below: Main table experiment(5/12), refer to paper Table 2
- env_name: frozen lake
- method: ragen w/o thinking
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
waiting to be done
```

[EXP 7] Below: Main table experiment(6/12), refer to paper Table 2
- env_name: bandit
- method: ragen w/o thinking
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
waiting to be done
```

[EXP 8] Below: Main table experiment(7/12), refer to paper Table 2
- env_name: sokoban
- method: sft
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
waiting to be done
```

[EXP 9] Below: Main table experiment(8/12), refer to paper Table 2
- env_name: frozen lake
- method: sft
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
waiting to be done
```

[EXP 10] Below: Main table experiment(9/12), refer to paper Table 2
- env_name: bandit
- method: sft
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
waiting to be done
```
[EXP 11] Below: Main table experiment(10/12), refer to paper Table 2
- env_name: sokoban
- method: prompt
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct

```bash
waiting to be done
```

[EXP 12] Below: Main table experiment(11/12), refer to paper Table 2
- env_name: frozen lake
- method: prompt
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
waiting to be done
```

[EXP 13] Below: Main table experiment(12/12), refer to paper Table 2
- env_name: bandit
- method: prompt
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
```bash
waiting to be done
```

[EXP 14] Below: Main table generalization experiment(1/3), refer to paper Figure 5
- method_name: ragen_trained_on_sokoban
- env_name: sokoban, frozen lake, bandit
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
- mode: inference mode
```bash
waiting to be done
```

[EXP 15] Below: Main table generalization experiment(2/3), refer to paper Figure 5
- method_name: ragen_trained_on_frozenlake
- env_name: sokoban, frozen lake, bandit
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
- mode: inference mode

```bash
waiting to be done
```

[EXP 16] Below: Main table generalization experiment(3/3), refer to paper Figure 5
- method_name: ragen_trained_on_bandit
- env_name: sokoban, frozen lake, bandit
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
- mode: inference mode

```bash
waiting to be done
```

[EXP 17] Below: ICL experiment(1/4), refer to paper Figure 6
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
- mode: train mode
- icl: 0

***Same as [EXP 2]***


[EXP 18] Below: ICL experiment(2/4), refer to paper Figure 6
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
- mode: train mode
- icl: 1
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=icl_sokoban_qwen_2.5_7b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True \
    training.prompt=[]
```

[EXP 19] Below: ICL experiment(3/4), refer to paper Figure 6
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
- mode: train mode
- icl: 2
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=icl_sokoban_qwen_2.5_7b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True \
    training.prompt=[]
```

[EXP 20] Below: ICL experiment(4/4), refer to paper Figure 6
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
- mode: train mode
- icl: 4
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=icl_sokoban_qwen_2.5_7b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True \
    training.prompt=[]
```

[EXP 21] Below: Model Scaling experiment(1/4), refer to paper Figure 7
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-0.5B-Instruct
- mode: train mode
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=model_scaling_sokoban_qwen_2.5_0.5b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True
```

[EXP 22] Below: Model Scaling experiment(2/4), refer to paper Figure 7
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-1.5B-Instruct
- mode: train mode
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-1.5B-Instruct \
    model.experiment_name=model_scaling_sokoban_qwen_2.5_1.5b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True
```

[EXP 23] Below: Model Scaling experiment(3/4), refer to paper Figure 7
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=model_scaling_sokoban_qwen_2.5_3b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True
```

[EXP 24] Below: Model Scaling experiment(4/4), refer to paper Figure 7
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-7B-Instruct
- mode: train mode

***Same as [EXP 2]***

[EXP 25] Below: Base vs Instruct, refer to paper Figure 7
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B
- mode: train mode
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B \
    model.experiment_name=base_vs_instruct_sokoban_qwen_2.5_3b_base \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True
```

[EXP 26] Below: Base vs Instruct, refer to paper Figure 8
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
***Same as [EXP 23]***

[EXP 27] RL algorithm (1/3), refer to paper table 3
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- RL algorithm: APO (which is PPO)
```bash
waiting to be done
```

[EXP 28] RL algorithm (2/3), refer to paper table 3
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- RL algorithm: BRPO
```bash
waiting to be done
```

[EXP 29] RL algorithm (3/3), refer to paper table 3
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- RL algorithm: GRPO
```bash
waiting to be done
```

[EXP 30] Below: context length extrapolation, refer to table 4
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- context length: 4000
***Same as [EXP 23]***

[EXP 31] Below: context length extrapolation, refer to table 4
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- context length: 8000
```bash
waiting to be done
```
[EXP 32] Below: context length extrapolation, refer to table 4
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- context length: 16000
```bash
waiting to be done
```

[EXP 33] Below: Do turns and temperature work? refer to paper Figure 9
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- temperature: 0
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=turns_and_temperature_sokoban_qwen_2.5_3b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True \
    training.temperature=0
```

[EXP 34] Below: Do turns and temperature work? refer to paper Figure 9
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- temperature: 0.5
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=turns_and_temperature_sokoban_qwen_2.5_3b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True \
    training.temperature=0.5
```

[EXP 35] Below: Do turns and temperature work? refer to paper Figure 9
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- temperature: 1

***Same as [EXP 23]***

[EXP 36] Below: Do turns and temperature work? refer to paper Figure 9
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- max_turns: 1
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=max_turns_sokoban_qwen_2.5_3b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True \
    training.max_turns=1
```

[EXP 37] Below: Do turns and temperature work? refer to paper Figure 9
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- max_turns: 5

***Same as [EXP 23]***

[EXP 38] Below: Do turns and temperature work? refer to paper Figure 9
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- max_turns: 8
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=max_turns_sokoban_qwen_2.5_3b_instruct \
    training.train_batch_size=[] \
    training.ppo_batch_size=[] \
    training.micro_batch_size=[] \
    optimization.adv_estimator=[] \
    training.use_kl_loss=True \
    training.max_turns=8
```

[EXP 39] Below: binary and non-binary reward, refer to paper Table 5
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- reward_type: binary
```bash
waiting to be done
    training.reward_type=binary
```

[EXP 40] Below: binary and non-binary reward, refer to paper Table 5
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- reward_type: non-binary
***Same as [EXP 23]***


[EXP 41] Below: Do we need to mask state? refer to paper Table 6
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- mask_state: True
```bash
waiting to be done
    training.mask_state=True
```

[EXP 42] Below: Do we need to mask state? refer to paper Table 6
- method_name: ragen
- env_name: sokoban
- rl_method: [blank]
- model_name: Qwen/Qwen2.5-3B-Instruct
- mode: train mode
- mask_state: False
***Same as [EXP 23]***


---
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
    model.experiment_name=test_zihan_brpo_p8r16 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128 \
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