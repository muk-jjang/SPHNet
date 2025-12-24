#!/bin/bash

# Multi-GPU batch MD17 evaluation script
# Processes multiple molecules in parallel across all available GPUs

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Default batch size = number of GPUs
BATCH_SIZE=${BATCH_SIZE:-$NUM_GPUS}

# Run batched evaluation
/data/miniconda3/envs/sphnet_gpueval/bin/python md17_evaluation_gpu_batch.py \
    --batch_size $BATCH_SIZE \
    "$@"

# Usage examples:
# ./run_evaluation_gpu_batch.sh --dir_path /path/to/data --size_limit 100
# BATCH_SIZE=4 ./run_evaluation_gpu_batch.sh --dir_path /path/to/data  # Use only 4 GPUs
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_evaluation_gpu_batch.sh --dir_path /path/to/data  # Use specific GPUs
