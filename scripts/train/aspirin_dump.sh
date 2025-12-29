#!/bin/bash

ROOT_DIR="/home/sungjun/repos/SPHNet"
cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1
python pipelines/test.py \
--config-name=aspirin.yaml \
job_id=aspirin \
save_output_dump=true \
inference_batch_size=1 \
ckpt_path=/nas/seongjun/sphnet/ \
log_dir=/nas/seongjun/sphnet/