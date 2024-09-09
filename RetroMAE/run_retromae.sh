#!/bin/bash
PORT_ID=$(expr $RANDOM + 1000)
# --max_steps 50000 \

ds="/app/snow.atg_arch_only.home/users/ariyaz/AI_Search/embedding_training_e5_flagembedding/e5_dataset/ms_marco_hn_mined_dataset_16june2024_v1.jsonl"
ds="/app/snow.riyaj_atar.home/ariyaz/AI_Search/embedding_training_e5_flagembedding/e5_dataset/embedding_dataset_merged_HN_gte_base.jsonl"
ds='/app/snow.riyaj_atar.home/ariyaz/AI_Search/GraphRank/translate_Task/retromae_ds.jsonl'

CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --master_port $PORT_ID --nproc_per_node 4 \
-m src.run \
--output_dir xlm_r_retromae_v100x4 \
--model_name_or_path 'FacebookAI/xlm-roberta-large'  \
--tokenizer_name 'FacebookAI/xlm-roberta-large' \
--train_data $ds \
--learning_rate 5e-6 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--dataloader_drop_last True \
--max_seq_length 512 \
--weight_decay 0.01 \
--save_steps 100 \
--report_to 'wandb' \
--lr_scheduler_type "cosine" \
--logging_steps 5 