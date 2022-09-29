#!/bin/bash

source .env/bin/activate
deepspeed run_clm.py \
    --deepspeed ds_config.json \
    --model_name_or_path output \
    --tokenizer_name juman-bpe-wiki-cc100-50000 \
    --train_file processed_dataset_50000 \
    --do_train \
    --output_dir /home/mdxuser/projects/gpt2-test/gpt2-xl-japanese2 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 22 \
    --learning_rate 3e-04 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --num_train_epochs 9 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --save_total_limit 1 \
    --bf16 \
    --logging_steps 4 \
    --push_to_hub \
    --hub_model_id schnell/gpt2-xl-japanese2 \
    --hub_strategy checkpoint \
    --hub_token ${HUB_TOKEN}
    
#初回学習時のパラメタ上書き設定(2回目以降はローカルに保存したパラメタを利用)
    # --config_overrides="vocab_size=50000,bos_token_id=1,eos_token_id=2,n_embd=1600,n_layer=48,n_head=20" \
