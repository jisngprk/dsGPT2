#! /bin/bash

python model_loader.py \
      --model_select 112m \
      --vocab_id_dir vocab_50257 \
	    --workspace finetune4 \
	    --ckpt_id epoch33-step14000 \
	    --enable_padding False \
	    --enable_bos True \
	    --enable_eos True \
	    --truncated_len 3 \
	    --repetition_penalty 1.5 \
	    --min_length 5 \
	    --max_length 10 \
	    --train_mode finetune \
	    --num_beams 5 \
	    --use_cpu True
