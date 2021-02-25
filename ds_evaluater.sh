#! /bin/bash

GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
	ds_evaluater.py \
	--load_dir ./checkpoints/test3 \
	--ckpt_id epoch7-step78000-loss4.06/ \
	--deepspeed \
	--deepspeed_config ./config/ds_config.json