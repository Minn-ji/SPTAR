import os

import torch
import json
import csv
import argparse
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from random import randint

from xuyang.default_prompt import DefaultPrompt

# 약한 쿼리(weak queries)'를 실제로 생성하는 역할
def simple_filter(text):
    text = text.split("\n")[0]
    for punt in [".", ",", "?"]:
        pre_i = ""
        new_text = ""
        for i in text.split(punt):
            if pre_i != i:
                new_text += i
                pre_i = i
            else:
                break
        text = new_text
    return text

def fix_prompt_controller(dataset_name):
    if 'ms_' in dataset_name:
        fixed_one_shot_prompt = DefaultPrompt.ms_50_fixed_one_shot_prompt
        fixed_two_shot_prompt = DefaultPrompt.ms_50_fixed_two_shot_prompt
    if 'law' in dataset_name:
        fixed_one_shot_prompt = DefaultPrompt.law_50_fixed_one_shot_prompt
        fixed_two_shot_prompt = DefaultPrompt.law_50_fixed_two_shot_prompt
    return fixed_one_shot_prompt, fixed_two_shot_prompt

def weak_infer(model, tokenizer, corpus_filtered, device, weak_queries_path, weak_train_path, prompt_num, train_prompt, dataset_name, new_query_id=2000000, fixed_prompt=True):
    os.makedirs(f'inference_output/{dataset_name}/', exist_ok=True)
    model.eval()
    weak_query_list = []
    weak_query_doc_id_list = []
    train_prompt_len = len(train_prompt)
    if fixed_prompt:
        fixed_one_shot_prompt, fixed_two_shot_prompt = fix_prompt_controller(dataset_name)
    for i in tqdm(corpus_filtered):
        # try:
        new_query_id += 1
        corpus_id = i['_id']
        corpus_text = i['text']
        gen_trail = 0
        while(gen_trail < 3):
            if int(prompt_num) == 2:
                if fixed_prompt:
                    new_prompt = "{} \n Document: {} \n Relevant Query: ".format(fixed_one_shot_prompt, corpus_text)
                else:
                    prompt_idx_1 = randint(0, train_prompt_len-1)
                    new_prompt = "Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".format(train_prompt[prompt_idx_1]['text_x'], train_prompt[prompt_idx_1]['text_y'], corpus_text)
            elif int(prompt_num) == 3:
                if fixed_prompt:
                    new_prompt = "{} \n Document: {} \n Relevant Query: ".format(fixed_two_shot_prompt, corpus_text)
                else:
                    prompt_idx_1 = randint(0, train_prompt_len-1)
                    prompt_idx_2 = randint(0, train_prompt_len-1)
                    new_prompt = "Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".format(train_prompt[prompt_idx_1]['text_x'], train_prompt[prompt_idx_1]['text_y'], train_prompt[prompt_idx_2]['text_x'], train_prompt[prompt_idx_2]['text_y'], corpus_text)
            else:
                new_prompt = "Document: {} \n Relevant Query: ".format(corpus_text)
            inputs = tokenizer(new_prompt, return_tensors="pt")
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"], 
                    max_new_tokens=128, 
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.7
                )
                new_query = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(new_prompt):]
                new_query = simple_filter(new_query)
            if len(new_query) == 0:
                gen_trail += 1
            else:
                gen_trail = 4
        weak_query_doc_id_list.append([new_query_id, corpus_id, 1])
        weak_query_list.append({"_id": str(new_query_id), "text": new_query, "metadata": {}})
        # except:
        #     pass
    with open(weak_queries_path, 'w') as outfile:
        for entry in weak_query_list:
            json.dump(entry, outfile)
            outfile.write('\n')
    with open(weak_train_path, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(["query-id", "corpus-id", "score"])
        for qd_i in weak_query_doc_id_list:
            tsv_output.writerow(qd_i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu device")
    parser.add_argument("--weak_queries_path", type=str, default="", help="tokenizer")
    parser.add_argument("--weak_train_path", type=str, default="", help="tokenizer")
    parser.add_argument("--peft_model_id", type=str, help="our model checkpoint")
    parser.add_argument("--data_path", type=str, help="data")
    parser.add_argument("--prompt_num", type=int, default=1, help="data")
    parser.add_argument("--train_prompt_path", type=str, help="train_prompt_path")
    parser.add_argument("--dataset_name", type=str, default="ms_50", help="dataset_name")
    parser.add_argument("--new_query_id", type=int, default=2000000, help="dataset_name")
    args = parser.parse_args()

    config = PeftConfig.from_pretrained(args.peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.to(args.device)
    corpus_filtered = json.loads(pd.read_csv(args.data_path).iloc[:10,:].to_json(orient="records"))
    train_prompt = json.loads(pd.read_csv(args.train_prompt_path).to_json(orient="records"))
    weak_infer(model, tokenizer, corpus_filtered, args.device, args.weak_queries_path, args.weak_train_path,
               args.prompt_num, train_prompt, args.dataset_name, int(args.new_query_id))



# python -m xuyang.weak_inference --weak_queries_path inference_output/law/weak_queries_50_tiny_llama-1.1b_523_prompt_3.jsonl \
#     --weak_train_path inference_output/law/weak_queries_50_tiny_llama-1.1b_523_prompt_3.csv \
#     --peft_model_id llm_models/v1_pointwise_without_prompt_example_law_tiny_llama-1.1b_tiny_llama-1.1b_CAUSAL_LM_TEXT_50_50_3_fixed_prompt_contractive_hard_10_val_loss_2025-08-09_1/tiny_llama-1.1b_CAUSAL_LM_TEXT/
#     --data_path xuyang/data/law/100k/corpus_filtered.csv \
#     --prompt_num 3 \
#     --dataset_name law \
#     --new_query_id 5000000 \
#     --train_prompt_path xuyang/data/law/prompt_tuning_train_text.csv \
#     --device cpu gpu일때 생략
