#!/bin/bash

# Always run from repo root regardless of call location
ROOT_DIR="/home/sungjun/repos/SPHNet"
cd "$ROOT_DIR" || exit 1

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

python md17_evaluation_customv2.py \
--dir_path=/nas/seongjun/sphnet/salicylic_acid/output_dump \
--num_procs=60 \
--size_limit=-1 \
--use_gpu=-1 \
--do_new_calc
    