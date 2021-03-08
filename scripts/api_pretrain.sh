#! /bin/bash

python api.py \
      --model_select 112m \
      --vocab_id_dir vocab_50257 \
	    --workspace pretrain \
	    --ckpt_id epoch1-step140000 \
	    --enable_padding False \
	    --enable_bos True \
	    --enable_eos False \
	    --truncated_len 128 \
	    --min_length 50 \
	    --max_length 70 \
	    --do_sample True \
	    --top_k 10 \
	    --temperature 0.9 \
      --repetition_penalty 1.5 \
	    --train_mode pretrain \
	    --num_beams 5 \
	    --use_cpu True \
	    --port 4000
