chmod +x $(pwd)/multi_run.py

# # fiqa
# CUDA_VISIBLE_DEVICES=2 python retrieve/dpr_contriever_eval.py --dataset_name fiqa --version v1 --gpu_id 0 --train_num 50 -exps no_aug --weak_num 100k
# # msmarco
# CUDA_VISIBLE_DEVICES=3 python retrieve/dpr_contriever_eval.py --dataset_name msmarco --version v1 --gpu_id 1 --train_num 50 -exps no_aug --weak_num 100k
# # fiqa
# CUDA_VISIBLE_DEVICES=4 python retrieve/dpr_contriever_eval.py --dataset_name fiqa --version v1 --gpu_id 2 --train_num 50 -exps p_written_100k_vicuna_prompt_2_filtered_70 --weak_num 100k
# # msmarco
# CUDA_VISIBLE_DEVICES=5 python retrieve/dpr_contriever_eval.py --dataset_name msmarco --version v1 --gpu_id 3 --train_num 50 -exps p_written_100k_vicuna_prompt_3_filtered_30 --weak_num 100k
# # fiqa
# CUDA_VISIBLE_DEVICES=6 python retrieve/dpr_contriever_eval.py --dataset_name fiqa --version v1 --gpu_id 4 --train_num 50 -exps llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 --weak_num 100k
# # msmarco
# CUDA_VISIBLE_DEVICES=7 python retrieve/dpr_contriever_eval.py --dataset_name msmarco --version v1 --gpu_id 5 --train_num 50 -exps llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30 --weak_num 100k

# fiqa
# CUDA_VISIBLE_DEVICES=1 python retrieve/bm25ce_eval.py --dataset_name fiqa --version v1 --gpu_id 1 --train_num 50 -exps no_aug --weak_num 100k
# msmarco
# CUDA_VISIBLE_DEVICES=2 python retrieve/bm25ce_eval.py --dataset_name msmarco --version v1 --gpu_id 2 --train_num 50 -exps no_aug --weak_num 100k
# fiqa
# CUDA_VISIBLE_DEVICES=3 python retrieve/bm25ce_eval.py --dataset_name fiqa --version v1 --gpu_id 3 --train_num 50 -exps p_written_100k_vicuna_prompt_2_filtered_70 --weak_num 100k
# msmarco
# CUDA_VISIBLE_DEVICES=4 python retrieve/bm25ce_eval.py --dataset_name msmarco --version v1 --gpu_id 4 --train_num 50 -exps p_written_100k_vicuna_prompt_3_filtered_30 --weak_num 100k
# fiqa
# CUDA_VISIBLE_DEVICES=6 python retrieve/bm25ce_eval.py --dataset_name fiqa --version v1 --gpu_id 6 --train_num 50 -exps llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 --weak_num 100k
# msmarco
# CUDA_VISIBLE_DEVICES=7 python retrieve/bm25ce_eval.py --dataset_name msmarco --version v1 --gpu_id 7 --train_num 50 -exps llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30 --weak_num 100k


parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--weak_num', required=False, default="5000", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
et_name msmarco --train_num 50 --exp_name llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30 --dpr_v v1