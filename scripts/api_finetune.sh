#! /bin/bash

python api.py \
      --model_select 112m \
      --vocab_id_dir vocab_50257 \
	    --workspace finetune5 \
	    --ckpt_id epoch82-step84000 \
	    --enable_padding False \
	    --enable_bos True \
	    --enable_eos True \
	    --truncated_len 122 \
	    --min_length 10 \
	    --max_length 30 \
	    --train_mode finetune \
	    --do_sample True \
	    --top_p 0.8 \
	    --temperature 0.9 \
      --repetition_penalty 1.5 \
	    --use_cpu True \
	    --port 4001
