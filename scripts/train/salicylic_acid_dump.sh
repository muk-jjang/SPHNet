#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=3
python pipelines/test.py \
--config-name=salicylic_acid.yaml \
save_output_dump=true \
inference_batch_size=216 \
ckpt_path=/nas/seongjun/sphnet/ \
log_dir=/nas/seongjun/sphnet/