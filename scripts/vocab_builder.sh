#! /bin/bash

python vocab_builder.py \
      --vocab_train_dir ./data_files/vocab_trains \
      --vocab_dir ./vocab/vocab_50257 \
      --vocab_prefix vocab_50257 \
      --vocab_size 50257 \
      --nspecial 100
