#!/bin/bash

# GPU-accelerated MD17 evaluation script
# This script sets up the CUDA environment and runs the evaluation with GPU support

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Run evaluation with GPU
# Single GPU mode by default, use --use_gpu with comma-separated IDs for multi-GPU
python md17_evaluation_customv2.py \
    --use_gpu 0 \
    "$@"

# Usage examples:
# Single GPU mode:
#   ./run_evaluation_gpu.sh --dir_path /path/to/data --size_limit 10
#   ./run_evaluation_gpu.sh --dir_path /path/to/data --size_limit 100 --debug
#
# Multi-GPU mode (use GPUs 0,1,2,3 with 4 processes):
#   python md17_evaluation_customv2.py --use_gpu 0,1,2,3 --num_procs 4 --dir_path /path/to/data
#
# Multi-GPU mode with all GPUs:
#   GPU_LIST=$(nvidia-smi --list-gpus | wc -l | xargs -I {} seq -s, 0 $(({}‐1)))
#   python md17_evaluation_customv2.py --use_gpu $GPU_LIST --num_procs $NUM_GPUS --dir_path /path/to/data
