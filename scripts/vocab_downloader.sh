#! /bin/bash

python vocab_downloader.py \
      --config_src ./config/db_config_vocab.json \
      --num_process 30 \
      --vocab_train_dir ./data_files/vocab_trains \
      --vocab_train_fname vocab_train \
      --nsample 50_000_000
