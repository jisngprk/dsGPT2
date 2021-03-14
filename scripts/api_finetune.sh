#! /bin/bash

python api.py \
      --model_select 112m \
      --vocab_id_dir vocab_50257_ns \
	    --workspace finetune8 \
	    --ckpt_id epoch76-step78000 \
	    --enable_padding False \
	    --enable_bos True \
	    --enable_eos True \
	    --truncated_len 122 \
	    --min_length 10 \
	    --max_length 50 \
	    --train_mode finetune \
	    --do_sample True \
	    --top_p 0.8 \
	    --temperature 0.6 \
      --repetition_penalty 1.2 \
	    --use_cpu True \
	    --port 4001
