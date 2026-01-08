#!/bin/bash

ROOT_DIR="/home/sungjun/repos/SPHNet/"

cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1

python pipelines/train.py --config-name=malondialdehyde.yaml \
job_id=malondialdehyde_transformed_lmdb_original \
data_name=escflow_malondialdehyde \
ckpt_path=./transformed_outputs \
log_dir=./transformed_outputs \
save_output_dump=true \
batch_size=10 \
inference_batch_size=32 \
dataset_path=/ssd1/qhflow-mlff/dataset/malondialdehyde_shard/processed/lmdbs \
dataloader_num_workers=1 \
devices=[0]