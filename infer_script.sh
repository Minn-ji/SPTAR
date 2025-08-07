#!/bin/bash

# zhiyuan 100k test, ms50 llama in-context 3 only infer fix, best_llama_prompt_3, for 5000
python -m xuyang.weak_inference --weak_queries_path xuyang/data/msmarco_50/100k/weak_queries_50_gpt2_100k_prompt_3.jsonl \
    --weak_train_path xuyang/data/msmarco_50/100k/weak_train_50_gpt2_100k_prompt_3.tsv \
    --peft_model_id llm_models/v1_pointwise_ms_50_gpt2_gpt2_CAUSAL_LM_TEXT_50_50_3_2025-08-07_0/gpt2_CAUSAL_LM_TEXT \
    --data_path xuyang/data/msmarco_50/100k/corpus_filtered_100k.csv \
    --prompt_num 3 \
    --dataset_name ms_50 \
    --new_query_id 5000000 \
    --train_prompt_path xuyang/data/msmarco_50/prompt_tuning_50_train_text.csv \
    --device cpu
#
## zhiyuan 5000 test, ms50 llama in-context 3 only infer fix, best_llama_prompt_3, for 100k_2
#python weak_inference.py \
#    --device cuda:6 \
#    --weak_queries_path /home/xwu/project/LLMsAgumentedIR/xuyang/data/msmarco_50/5000/weak_queries_500_llama-7b_5000_llama_prompt_3.jsonl \
#    --weak_train_path /home/xwu/project/LLMsAgumentedIR/xuyang/data/msmarco_50/5000/weak_train_500_llama-7b_5000_llama_prompt_3.tsv \
#    --peft_model_id /home/xwu/project/SPTAR/xuyang/llm_models/v1_msmarco_50_llama-7b_llama-7b_CAUSAL_LM_TEXT_50_500_3_2024-08-23_0/llama-7b_CAUSAL_LM_TEXT \
#    --data_path /home/xwu/project/SPTAR/xuyang/data/msmarco_50/5000/corpus_filtered_5000.csv \
#    --prompt_num 3 \
#    --dataset_name ms_50 \
#    --new_query_id 5000000 \
#    --train_prompt_path /home/xwu/project/SPTAR/xuyang/data/msmarco_50/prompt_tuning_500_train_text_sampled.csv
#
## zhiyuan 5000 test, ms50 llama in-context 3 only infer fix, best_llama_prompt_3, for 100k_2
#python weak_inference.py \
#    --device cuda:1 \
#    --weak_queries_path /home/xwu/project/LLMsAgumentedIR/xuyang/data/msmarco_50/5000/weak_queries_1000_llama-7b_5000_llama_prompt_3.jsonl \
#    --weak_train_path /home/xwu/project/LLMsAgumentedIR/xuyang/data/msmarco_50/5000/weak_train_1000_llama-7b_5000_llama_prompt_3.tsv \
#    --peft_model_id /home/xwu/project/SPTAR/xuyang/llm_models/v1_msmarco_50_llama-7b_llama-7b_CAUSAL_LM_TEXT_50_1000_3_2024-08-23_0/llama-7b_CAUSAL_LM_TEXT \
#    --data_path /home/xwu/project/SPTAR/xuyang/data/msmarco_50/5000/corpus_filtered_5000.csv \
#    --prompt_num 3 \
#    --dataset_name ms_50 \
#    --new_query_id 5000000 \
#    --train_prompt_path /home/xwu/project/SPTAR/xuyang/data/msmarco_50/prompt_tuning_1000_train_text_sampled.csv