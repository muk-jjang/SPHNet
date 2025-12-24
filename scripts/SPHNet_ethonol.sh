#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

python pipelines/train.py --config-name=ethanol.yaml \
job_id=escflow_ethanol_new \
ckpt_path=/nas/seongjun/sphnet/outputs2 \
log_dir=/nas/seongjun/sphnet/outputs2 \
save_output_dump=true \
batch_size=10 \
inference_batch_size=32 \
devices=[1]