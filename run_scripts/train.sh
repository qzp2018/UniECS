#!/usr/bin/env
# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# Command: bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}
# Number of GPUs per GPU worker
GPUS_PER_NODE=2
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=855
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip/
DATAPATH=""
experiment_name=''
# 新增：打印DATAPATH和training_history.npy的存储路径
echo "===== 路径信息 ====="
echo "DATAPATH（输入数据根目录）: ${DATAPATH}"
#/home/daihuangyu/.jupyter/lxy/clip/data
# data options
train_data=train_data_file_address
val_data=test_data_file_address
# restore options
resume=customed_ckpt #our customed ckpt path to resume
reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
# reset_optimizer=""
# output options
output_base_dir=${DATAPATH}/experiments/${experiment_name}
name=model_name
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=50
# training hyper-params
context_length=64
warmup=2000
batch_size=16
valid_batch_size=200
accum_freq=8
lr=1e-6
wd=0.01
alignment_weight=0.5
v2t_weight=1.0
v2f_weight=1.0
t2f_weight=0.2
vision_lr=1e-5
v2v_temperature=0.05

v2v_contra_weight=2.0
max_epochs=50  # or you can alternatively specify --max-steps
valid_step_interval=500
valid_epoch_interval=3
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
use_augment="--use-augment"

python3 -m torch.distributed.launch --use_env --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
          -m src.training.main \
          --train-data=${train_data} \
          --val-data=${val_data} \
          --resume=${resume} \
          ${reset_data_offset} \
          ${reset_optimizer} \
          --logspace=${output_base_dir} \
          --name=${name} \
          --save-step-frequency=${save_step_frequency} \
          --save-epoch-frequency=${save_epoch_frequency} \
          --log-interval=${log_interval} \
          --context-length=${context_length} \
          --warmup=${warmup} \
          --batch-size=${batch_size} \
          --valid-batch-size=${valid_batch_size} \
          --valid-step-interval=${valid_step_interval} \
          --valid-epoch-interval=${valid_epoch_interval} \
          --accum-freq=${accum_freq} \
          --lr=${lr} \
          --wd=${wd} \
          --vision-lr=${vision_lr} \
          --v2v-temperature=${v2v_temperature} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          ${use_augment} \
          --text-model=${text_model} \
          --alignment_weight=${alignment_weight} \
          --v2t_weight=${v2t_weight} \
          --v2f_weight=${v2f_weight} \
          --t2f_weight=${t2f_weight} \
          --v2v_contra_weight=${v2v_contra_weight} \
          --text-model=${text_model}