#! /bin/bash

python vocab_builder.py \
      --vocab_train_dir ./data_files/vocab_trains_no_social \
      --vocab_dir ./vocab/vocab_50257_ns \
      --vocab_prefix vocab_50257_ns \
      --vocab_size 50257 \
      --nspecial 100
