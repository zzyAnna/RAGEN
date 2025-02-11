#!/bin/bash

# Default environment
ENV_NAME=${1:-"sokoban"}

# Get and execute the training command
python ragen/train.py "$ENV_NAME" "$@" | bash