# preprocessed_datasetという名でデータセットを前処理する

python /home/mdxuser/projects/gpt2-test/preprocess_datasets.py \
    --model_type gpt2 \
    --tokenizer_name juman-bpe-wiki-cc100-50000 \
    --train_file train.txt \
    --config_overrides="vocab_size=50000,bos_token_id=1,eos_token_id=2" \
    --do_train \
    --output_dir /home/mdxuser/projects/gpt2-test/output2 \
    --validation_split_percentage 1 \
    --preprocessing_num_workers 10 \
