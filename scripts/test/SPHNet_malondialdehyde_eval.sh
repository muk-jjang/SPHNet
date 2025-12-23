#!/bin/bash

# Always run from repo root regardless of call location
ROOT_DIR="/home/sungjun/repos/SPHNet"
cd "$ROOT_DIR" || exit 1

# Use direct Python path from conda environment
PYTHON_PATH="/home/chanhui-lee/.conda/envs/sphnet-gpu4pyscf/bin/python"

export HYDRA_FULL_ERROR=1

"$PYTHON_PATH" md17_evaluation_customv2.py \
--dir_path=/nas/seongjun/sphnet/malondialdehyde/output_dump \
--num_procs=1 \
--size_limit=-1 \
--use_gpu=1
