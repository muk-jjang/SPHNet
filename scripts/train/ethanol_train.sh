export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

ROOT_DIR="/home/sungjun/repos/SPHNet"
cd "$ROOT_DIR" || exit 1
export HYDRA_FULL_ERROR=1

devices=$1
dataset_name=ethanol
python pipelines/train.py \
--config-name=${dataset_name}.yaml \
save_output_dump=true \
job_id=${dataset_name}-sphnet_transform_optimized \
inference_batch_size=128 \
dataset_path=/ssd1/qhflow-mlff/dataset/ethanol_shard/processed/lmdbs \
dataloader_num_workers=0 \
devices=${devices} \
model=sphnet-36M \
wandb.wandb_api_key=${WANDB_API_KEY_KU_AI4SIM}