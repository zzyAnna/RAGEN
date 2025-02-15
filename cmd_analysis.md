# 

3.1 Experiments
```bash
bash scripts/analysis/bandits.sh



















bash train.sh sokoban \
    model.experiment_name=sokoban_main \
    system.cuda_visible_devices=0 \
    training.micro_batch_size=2 \
    training.total_training_steps=100 \
    trainer.test_freq=25  >> ./log/terminal/sokoban_main.log


bash train.sh frozenlake \
    model.experiment_name=sokoban_main \
    system.cuda_visible_devices=2 \
    training.micro_batch_size=2 \
    training.total_training_steps=100 \
    trainer.test_freq=25  >> ./log/terminal/frozenlake_main.log


```