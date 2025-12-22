#!/bin/bash

# Always run from repo root regardless of call location
ROOT_DIR="/home/sungjun/repos/SPHNet"
cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
python pipelines/train.py --config-name=malondialdehyde.yaml \
job_id=malondialdehyde_batch64_mae \
wandb.wandb_project=SPHNet_with_chlee \
ckpt_path=/nas/seongjun/sphnet/outputs \
log_dir=/nas/seongjun/sphnet/outputs \
save_output_dump=true \
batch_size=64 \
max_steps=300000 \
ngpus=2 \
devices=[0,4] \
hami_train_loss=mae \
inference_batch_size=64 
    