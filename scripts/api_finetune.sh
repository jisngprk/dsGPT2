#! /bin/bash

python api.py \
      --model_select 112m \
      --vocab_id_dir vocab_50257 \
	    --workspace finetune5 \
	    --ckpt_id epoch9-step10000 \
	    --enable_padding False \
	    --enable_bos True \
	    --enable_eos True \
	    --truncated_len 122 \
	    --min_length 20 \
	    --max_length 25 \
	    --train_mode finetune \
	    --num_beams 3 \
	    --use_cpu True \
	    --port 4001
