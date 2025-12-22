#!/bin/bash

# GPU-accelerated MD17 evaluation script
# This script sets up the CUDA environment and runs the evaluation with GPU support

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Run evaluation with GPU
python md17_evaluation_customv2.py \
    --use_gpu \
    --num_procs 5 \
    --dir_path /nas/seongjun/sphnet/malondialdehyde/output_dump \
    "$@"

# Usage examples:
# ./run_evaluation_gpu.sh --dir_path /path/to/data --size_limit 10
# ./run_evaluation_gpu.sh --dir_path /path/to/data --size_limit 100 --debug
