#!/bin/bash
PORT_ID=$(expr $RANDOM + 1000)
ds="/app/snow.atg_arch_only.home/users/ariyaz/AI_Search/embedding_training_e5_flagembedding/Untitled Folder 1/FlagEmbedding/embedding_dataset_merged_HN.jsonl"
ds0="/app/snow.atg_arch_only.home/users/ariyaz/AI_Search/embedding_training_e5_flagembedding/e5_dataset/ms_marco_hn_mined_dataset_16june2024_v1.jsonl"

# Allow multiple threads
# export OMP_NUM_THREADS=16
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --master_port $PORT_ID --nproc_per_node 1 \
-m src.run \
--output_dir "multivector-xlm_r_large" \
--model_name_or_path 'FacebookAI/xlm-roberta-large' \
--train_data "/app/snow.riyaj_atar.home/ariyaz/AI_Search/embedding_training_e5_flagembedding/e5_dataset/embedding_dataset_merged_HN.jsonl" \
--learning_rate 5e-4 \
--fp16 \
--max_steps 250000 \
--per_device_train_batch_size 32 \
--normlized True \
--temperature 0.02 \
--query_max_len 256 \
--passage_max_len 512 \
--train_group_size 8 \
--negatives_cross_device \
--logging_steps 5 \
--gradient_checkpointing \
--gradient_accumulation_steps 2 \
--lr_scheduler_type "cosine"