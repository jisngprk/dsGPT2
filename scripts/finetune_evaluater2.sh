#! /bin/bash

python model_loader.py \
      --model_select 112m \
      --vocab_id_dir vocab_50257 \
	    --workspace fintune0 \
	    --ckpt_id epoch20-step2000 \
	    --enable_padding False \
	    --enable_bos True \
	    --enable_eos True \
	    --repetition_penalty 1.8 \
	    --min_length 20 \
	    --max_length 50 \
	    --train_mode finetune \
	    --use_cpu True