#!/bin/bash

# Always run from repo root regardless of call location
ROOT_DIR="/home/sungjun/repos/SPHNet"
cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2
python md17_evaluation_customv2.py \
--dir_path=/nas/seongjun/sphnet/salicylic_acid/output_dump_batch \
--num_procs=1 \
--debug \
--size_limit=1 \
    