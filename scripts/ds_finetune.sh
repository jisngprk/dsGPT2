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
	--enable_padding False \
	--enable_bos True \
	--enable_eos True \
	--truncated_len 122 \
	--wandb_dir kg_gpt2_finetune \
	--model_select 112m \
	--train_mode finetune \
	--workspace finetune8 \
	--alpha 0.0 \
	--workspace_finetune pretrain0314 \
	--ckpt_id_finetune epoch1-step142000 \
	--vocab_id_dir vocab_50257_ns \
	--eval_batch_size 128 \
	--train_iters 150000 \
	--ckpt_save_steps 2000 \
	--config_train ./config/db_config_finetune.json \
	--deepspeed \
	--deepspeed_config ./config/ds_config_finetune.json
