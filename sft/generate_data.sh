export DIM_X=6
export DIM_Y=6
export NUM_BOXES=1
export MAX_STEPS=5
export SEARCH_DEPTH=30

python sft/utils/generate_sft_verl.py \
    --env sokoban \
    --algo bfs \
    --seed 100000 \
    --output sft/data \
    --train_size 10000 \
    --test_size 100 \
    --bfs_max_depths 100 \
    --prefix message \
    --num_processes 16