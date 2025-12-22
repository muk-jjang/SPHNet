#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1
python pipelines/train.py --config-name=malondialdehyde.yaml \
wandb_project=SPHNet_with_chlee \
ckpt_path=/nas/seongjun/sphnet/malondialdehyde \
log_dir=/nas/seongjun/sphnet/malondialdehyde \
save_output_dump=true \
batch_size=64 \
ngpus=2 \
devices=[0,1] \

    