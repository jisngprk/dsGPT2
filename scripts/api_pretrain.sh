#! /bin/bash

python api.py \
      --model_select 112m \
      --vocab_id_dir vocab_50257_ns \
	    --workspace pretrain0314 \
	    --ckpt_id epoch1-step142000 \
	    --enable_padding False \
	    --enable_bos True \
	    --enable_eos False \
	    --truncated_len 128 \
	    --min_length 50 \
	    --max_length 128 \
	    --do_sample True \
	    --top_p 0.9 \
	    --temperature 0.8 \
      --repetition_penalty 1.2 \
	    --train_mode pretrain \
	    --use_cpu True \
	    --port 4000
