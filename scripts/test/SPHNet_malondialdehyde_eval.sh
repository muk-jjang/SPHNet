#!/bin/bash

# Always run from repo root regardless of call location
ROOT_DIR="/home/sungjun/repos/SPHNet"
cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1
python md17_evaluation_custom.py \
--dir_path=./outputs/malondialdehyde_split_25000_500_1478_pbe0/output_dump \
--num_procs=64 \
    