#!/bin/bash

ROOT_DIR="/home/sungjun/repos/SPHNet"

cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1

python pipelines/train.py --config-name=ethanol.yaml \
job_id=ethanol_lmdb_transformed \
ckpt_path=./transformed_outputs \
log_dir=./transformed_outputs \
save_output_dump=true \
batch_size=10 \
inference_batch_size=32 \
data_name=ethanol \
dataset_path=/ssd1/qhflow-mlff/dataset/ \
dataloader_num_workers=8 \
devices=[0] 