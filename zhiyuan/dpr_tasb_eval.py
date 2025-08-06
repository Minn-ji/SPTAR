#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   dpr_contriever_eval.py
@Time    :   2024/05/25 15:40:50
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   copy from dpr_eval, chnage it to dpr_tasb
'''

import subprocess
import argparse


def multirun(args):
    arg_dict = {}
    for key, vals in args._get_kwargs():
        arg_dict[key] = vals
    for exp in arg_dict["exp_names"]:
        print(f"GPU {arg_dict['gpu_id']} Training: {arg_dict['dataset_name']} on {exp}")
        if arg_dict["version"] == "v1":
            train_command = ["python", "zhiyuan/retriever/dpr_tasb/train/train_tasb.py", "--dataset_name", f"{arg_dict['dataset_name']}", "--train_num", f"{arg_dict['train_num']}", "--weak_num", f"{arg_dict['weak_num']}", "--exp_name", exp]
        # for test
        print(" ".join(train_command))
        print(f"GPU {arg_dict['gpu_id']} Training: {arg_dict['dataset_name']} on {exp}")
        subprocess.call(train_command)
        eval_command = ["python", "zhiyuan/retriever/dpr_tasb/eval/evaluate_tasb.py", "--dataset_name", f"{arg_dict['dataset_name']}", "--train_num", f"{arg_dict['train_num']}", "--exp_name", exp, "--dpr_v", arg_dict["version"]]
        # for test
        print(" ".join(eval_command))
        print("\n")
        subprocess.call(eval_command)

def main():
    parser = argparse.ArgumentParser(description='Training Starts ...')
    parser.add_argument("--dataset_name", type=str, help="")
    parser.add_argument("--version", type=str, help="")
    parser.add_argument("--gpu_id", type=int, help="")
    parser.add_argument("--train_num", type=int, help="")
    parser.add_argument("--weak_num", type=str, help="")
    parser.add_argument('-exps','--exp_names', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    multirun(args)

if __name__ =="__main__":
    main()
    '''
    python zhiyuan/dpr_tasb_eval.py \
    --dataset_name msmarco \
    --version v1 \
    --gpu_id 0 \
    --train_num 50 \
    --weak_num 100k \
    --exp_names no_aug p_written_100k_vicuna_prompt_2_filtered_70 llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
'''