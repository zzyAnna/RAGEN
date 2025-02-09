#!/bin/bash

# Default environment
ENV_NAME=${ENV_NAME:-"sokoban"}

# Get and execute the training command
python ragen/train.py "$ENV_NAME" "$@" | bash