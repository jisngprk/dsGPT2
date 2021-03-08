#! /bin/bash

python model_loader.py \
      --model_select 112m \
      --vocab_id_dir vocab_50257 \
	    --workspace test25 \
	    --ckpt_id epoch1-step144000 \
	    --enable_padding False \
	    --enable_bos True \
	    --enable_eos False \
	    --repetition_penalty 1.8 \
	    --use_cpu True
