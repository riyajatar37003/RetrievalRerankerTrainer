#!/bin/bash
PORT_ID=$(expr $RANDOM + 1000)

p='/app/snow.riyaj_atar.home/ariyaz/AI_Search/embedding_training_e5_flagembedding/e5_multilingual_ft_model/bert_multilingual_as_reranker'
ds="/app/snow.atg_arch_only.home/users/ariyaz/AI_Search/embedding_training_e5_flagembedding/e5_dataset/ms_marco_hn_mined_dataset_16june2024_v1.jsonl"

CUDA_VISIBLE_DEVICES=0, TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --master_port $PORT_ID --nproc_per_node 0 \
-m src.run \
--output_dir xlm_r_reranker \
--model_name_or_path $p  \
--teacher_model_path "/app/snow.riyaj_atar.home/ariyaz/AI_Search/embedding_training_e5_flagembedding/e5_multilingual_ft_model/best_gte_reranker_1" \
--student_tokenizer_path $p \
--train_data $ds \
--learning_rate 5e-6 \
--fp16 \
--max_steps 50000 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--dataloader_drop_last True \
--train_group_size 16 \
--max_len 512 \
--weight_decay 0.01 \
--save_steps 100 \
--report_to 'wandb' \
--lr_scheduler_type "linear" \
--logging_steps 5 