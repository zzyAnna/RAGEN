export DIM_X=6
export DIM_Y=6
export NUM_BOXES=2
export MAX_STEPS=10
export SEARCH_DEPTH=100

python sft/generate_sft_verl.py \
    --env sokoban \
    --algo bfs \
    --seed 100000 \
    --output sft/data \
    --train_size 10 \
    --test_size 10 \
    --bfs_max_depths 100 \
    --prefix message \
    --num_processes 16