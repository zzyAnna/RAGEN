mkdir -p log/terminal

# bash train.sh sokoban \
#     model.experiment_name=sokoban_main \
#     system.cuda_visible_devices=0 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_main.log &

# bash train.sh sokoban \
#     model.experiment_name=sokoban_main_no_think_rl \
#     system.cuda_visible_devices=1 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.no_think_rl=True \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_main_no_think_rl.log &

    
# bash train.sh frozenlake \
#     model.experiment_name=frozenlake_main \
#     system.cuda_visible_devices=2 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     trainer.test_freq=10  >> ./log/terminal/frozenlake_main.log &


# bash train.sh frozenlake \
#     model.experiment_name=frozenlake_main_no_think_rl \
#     system.cuda_visible_devices=3 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.no_think_rl=True \
#     trainer.test_freq=10  >> ./log/terminal/frozenlake_main_no_think_rl.log &





bash train.sh sokoban \
    model.experiment_name=sokoban_arpo \
    system.cuda_visible_devices=0 \
    training.micro_batch_size=2 \
    training.total_training_steps=100 \
    optimization.adv_estimator=arpo \
    trainer.test_freq=10  >> ./log/terminal/sokoban_arpo.log &

bash train.sh sokoban \
    model.experiment_name=sokoban_brpo \
    system.cuda_visible_devices=1 \
    training.micro_batch_size=2 \
    training.total_training_steps=100 \
    optimization.adv_estimator=brpo \
    trainer.test_freq=10  >> ./log/terminal/sokoban_brpo.log &

    
bash train.sh frozenlake \
    model.experiment_name=frozenlake_arpo \
    system.cuda_visible_devices=2 \
    training.micro_batch_size=2 \
    training.total_training_steps=100 \
    optimization.adv_estimator=arpo \
    trainer.test_freq=10  >> ./log/terminal/frozenlake_arpo.log &


bash train.sh frozenlake \
    model.experiment_name=frozenlake_brpo \
    system.cuda_visible_devices=3 \
    training.micro_batch_size=2 \
    training.total_training_steps=100 \
    optimization.adv_estimator=brpo \
    trainer.test_freq=10  >> ./log/terminal/frozenlake_brpo.log &
