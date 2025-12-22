#!/bin/bash

ROOT_DIR="/home/sungjun/repos/SPHNet"
cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
python pipelines/train.py --config-name=naphthalene.yaml \
job_id=naphthalene_batch10_step20 \
wandb.wandb_project=SPHNet \
ckpt_path=/nas/seongjun/sphnet/outputs \
log_dir=/nas/seongjun/sphnet/outputs \
save_output_dump=true \
batch_size=10 \
max_steps=200000 \
ngpus=1 \
devices=[0] \
inference_batch_size=32