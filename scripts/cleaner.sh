#! /bin/bash

python cleaner.py \
      --vocab_id_dir vocab_50257 \
      --enable_padding False \
      --enable_bos False \
      --enable_eos False \
      --config_src ./config/db_config_single_conv.json \
      --config_trgt ./config/db_config_filt_single_conv.json \
