#! /bin/bash

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
	ds_trainer.py \
	--wandb_dir kg_gpt2_server23 \
	--model_select 112m \
	--vocab_id_dir vocab_50257 \
	--workspace test0 \
	--eval_batch_size 128 \
	--train_iters 150000 \
	--config_train ./config/db_config_train.json \
	--deepspeed \
	--deepspeed_config ./config/ds_config.json
