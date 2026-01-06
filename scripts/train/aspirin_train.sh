devices=$1
dataset_name=aspirin
python pipelines/train.py \
--config-name=${dataset_name}.yaml \
save_output_dump=true \
job_id=${dataset_name}-36M \
inference_batch_size=128 \
ckpt_path=/nas/seongjun/sphnet \
log_dir=/nas/seongjun/sphnet \
devices=${devices} \
model=sphnet-36M \
wandb.wandb_api_key=${WANDB_API_KEY_KU_AI4SIM}