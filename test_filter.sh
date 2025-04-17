set -e


python train.py system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-base-rollout-filter-0-5-max-mean actor_rollout_ref.rollout.rollout_filter_ratio=0.5 actor_rollout_ref.rollout.rollout_filter_type=max_mean &
python train.py system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-base-rollout-filter-0-5-max-mean actor_rollout_ref.rollout.rollout_filter_ratio=0.5 actor_rollout_ref.rollout.rollout_filter_type=max_mean &
python train.py system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-base-rollout-filter-0-5-std actor_rollout_ref.rollout.rollout_filter_ratio=0.5 actor_rollout_ref.rollout.rollout_filter_type=std &
python train.py system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-base-rollout-filter-0-5-std actor_rollout_ref.rollout.rollout_filter_ratio=0.5 actor_rollout_ref.rollout.rollout_filter_type=std &

wait

python train.py system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-base-rollout-filter-0-25-max-mean actor_rollout_ref.rollout.rollout_filter_ratio=0.25 actor_rollout_ref.rollout.rollout_filter_type=max_mean &
python train.py system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-base-rollout-filter-0-25-max-mean actor_rollout_ref.rollout.rollout_filter_ratio=0.25 actor_rollout_ref.rollout.rollout_filter_type=max_mean &
python train.py system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-base-rollout-filter-0-25-std actor_rollout_ref.rollout.rollout_filter_ratio=0.25 actor_rollout_ref.rollout.rollout_filter_type=std &
python train.py system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-base-rollout-filter-0-25-std actor_rollout_ref.rollout.rollout_filter_ratio=0.25 actor_rollout_ref.rollout.rollout_filter_type=std &

wait

python train.py system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-base-rollout-filter-1 actor_rollout_ref.rollout.rollout_filter_ratio=1 &
python train.py system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-base-rollout-filter-1 actor_rollout_ref.rollout.rollout_filter_ratio=1 &
python train.py system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-base-rollout-filter-1 actor_rollout_ref.rollout.rollout_filter_ratio=1 &
python train.py system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-base-rollout-filter-1 actor_rollout_ref.rollout.rollout_filter_ratio=1 &

wait
