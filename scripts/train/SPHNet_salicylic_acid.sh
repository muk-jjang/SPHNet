#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2
python pipelines/train.py \
--config-name=salicylic_acid.yaml \
save_output_dump=true