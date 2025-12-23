#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1

python pipelines/train.py --config-name=malondialdehyde.yaml \
job_id=escflow_malondialdehyde_new \
ckpt_path=outputs2 \
log_dir=outputs2 \
save_output_dump=true \
batch_size=5 \
inference_batch_size=32 \
devices=[1]