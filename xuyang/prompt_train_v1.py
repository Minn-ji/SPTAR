from transformers import AutoModelForCausalLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import os
import wandb
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import argparse
from dotenv import load_dotenv
from xuyang.args import PromptTuringArgs
from xuyang.utils import reset_args, AverageMeter, setup_train
from xuyang.dataset import MSMARCODataset

load_dotenv()

hugging_face_api_key = os.getenv("HUGGINGFACE_API_KEY")

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
def _sanitize_and_check(batch, model, tokenizer):
    # dtype 보정
    if batch["labels"].dtype != torch.long:
        batch["labels"] = batch["labels"].long()

    # 라벨 보정: 음수 전부 -100로, vocab_size 이상도 -100
    with torch.no_grad():
        labels = batch["labels"]
        labels[labels < 0] = -100
        labels[labels >= model.config.vocab_size] = -100
        batch["labels"] = labels

    # input_ids 범위 검사 (임베딩 인덱스 오류 방지)
    ids = batch["input_ids"]
    V = model.config.vocab_size
    if not torch.all((ids >= 0) & (ids < V)):
        # 어디서 터졌는지 바로 알 수 있게 예외
        mn = int(ids.min().item()); mx = int(ids.max().item())
        raise RuntimeError(f"input_ids out of range: min={mn}, max={mx}, vocab_size={V}")

    return batch

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=hugging_face_api_key)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def main(args):
    # config
    export_root, args = setup_train(args)
    tokenizer = load_tokenizer(args)
    wandb.init(project="prompt-tuning-law", name=f"{args.llm_name}_{args.dataset_name}", config=args.__dict__)
    # dataset
    ir_dataset = MSMARCODataset(args, tokenizer)
    train_dataset, test_dataset = ir_dataset.get_dataset()

    train_dataloader = DataLoader(train_dataset['train'], shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(train_dataset['test'], collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)

    # creating model
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        # num_virtual_tokens=args.num_virtual_tokens 설정을 보고,
        # LLM의 임베딩 차원과 동일한 크기를 가진 num_virtual_tokens 개수만큼의 새로운 torch.nn.Parameter 객체
        # 즉, Soft Prompt의 파라미터를 생성함
        num_virtual_tokens=args.num_virtual_tokens,
        #prompt_tuning_init=PromptTuningInit.TEXT와 prompt_tuning_init_text=args.prompt_tuning_init_text 설정을 보고,
        # args.prompt_tuning_init_text에 지정된 텍스트(예: "please generate query for document" 반복)를
        # LLM의 원래 토크나이저와 임베딩 레이어를 사용하여 임베딩 벡터로 변환
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=args.prompt_tuning_init_text,
        tokenizer_name_or_path=args.model_name_or_path,
    )
    #  model은 LLM의 모든 파라미터를 포함(기존 임베딩, 모델 내부 파라미터 등)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 token=hugging_face_api_key,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto")

    # get_peft_model를 통해, 내부적으로 로드된 model의 파라미터을 처리함
    # LLM의 대부분의 원래 파라미터 전체, 즉 임베딩 및 모델 내부 파라미터를 requires_grad=False로 설정하여 freeze
    # peft_config에 정의된 새로운 Prompt Tuning 파라미터인 Soft Prompt 임베딩 레이어의 파라미터)만
    # requires_grad=True로 설정하여 학습 가능하게 함
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    # model.print_trainable_parameters()

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # training and evaluation
    model = model.to(args.device)
    original_eval_loss = 999999
    early_stop_epoch = 0
    for epoch in range(args.num_epochs):
        if early_stop_epoch > 5:
            print('Terminating because of early stopping!')
            break
        total_train_loss = 0
        total_eval_loss = 0
        avg_train_loss = AverageMeter()
        avg_val_loss = AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            batch = _sanitize_and_check(batch, model, tokenizer)
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                total_train_loss += loss.detach().float()
                avg_train_loss.update(loss.detach().float().item())
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # evaluate eval dataset
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            batch = _sanitize_and_check(batch, model, tokenizer)
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.detach().float()
            avg_val_loss.update(loss.detach().float().item())
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        # get metrics
        train_epoch_loss = total_train_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss).detach().float().item()
        eval_epoch_loss = total_eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss).detach().float().item()
        # saving model
        if avg_val_loss.avg < original_eval_loss:
            original_eval_loss = avg_val_loss.avg
            early_stop_epoch = 0
            filepath = Path(export_root).joinpath(args.peft_model_id)
            print(f'new best val loss, model saved at {filepath}')
            model.save_pretrained(filepath)
        else:
            early_stop_epoch += 1
        # logger
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss.avg,
            'val_loss': avg_val_loss.avg,
            'train_ppl': train_ppl,
            'eval_ppl': eval_ppl
        })
        print(f"{epoch=}: {train_ppl=} {avg_train_loss.avg=} {eval_ppl=} {avg_val_loss.avg=}")

    print('-------- 평가 --------')
    # evaluate test dataset
    total_test_loss = 0
    avg_test_loss = AverageMeter()
    model.eval()
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        batch = _sanitize_and_check(batch, model, tokenizer)
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_test_loss += loss.detach().float()
        avg_test_loss.update(loss.detach().float().item())
        preds = torch.argmax(outputs.logits, dim=-1)

        labels = batch['labels'].clone().detach().cpu()
        labels[labels == -100] = tokenizer.pad_token_id

        decoded_preds = tokenizer.batch_decode(preds.detach().cpu(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for i in range(len(decoded_preds)):
            print(f"\n[샘플 {i + 1}]")
            print("🔹 모델 생성:", decoded_preds[i])
            print("🔸 정답 레이블:", decoded_labels[i])
    test_epoch_loss = total_test_loss / len(test_dataloader)
    test_ppl = torch.exp(test_epoch_loss).detach().float().item()
    wandb.log({
        'test_loss': avg_test_loss.avg,
        'test_ppl': test_ppl
    })
    print(f"{epoch=}: {train_ppl=} {avg_train_loss.avg=} {eval_ppl=} {avg_val_loss.avg=} {test_ppl=} {avg_test_loss.avg=}")
    wandb.finish()

if __name__ == "__main__":
    base_args = PromptTuringArgs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_virtual_tokens", type=int, help="num virtual tokens for prompt")
    parser.add_argument("--llm_name", type=str, help="model name")
    parser.add_argument("--device_idx", type=str, help="device id")
    parser.add_argument("--prompt_num", type=int, help="prompt number")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--train_data", type=str, help="train data path")
    parser.add_argument("--eval_data", type=str, help="eval data path")
    parser.add_argument("--test_data", type=str, help="test data path")
    parser.add_argument("--few_shot_num", type=int, help="few shot setting")
    args = parser.parse_args(namespace=base_args)
    args = reset_args(args)
    main(args)
# 위치는 최상단 (SPTAR 바로 아래)
# --train_data xuyang/data/law/prompt_tuning_50_train_text.csv --eval_data xuyang/data/msmarco_50/prompt_tuning_50_test_text.csv --test_data xuyang/data/msmarco_50/prompt_tuning_50_test_text.csv
# python -m xuyang.prompt_train_v1 --device_idx 1 --num_virtual_tokens 50 --prompt_num 3 --llm_name tiny_llama-1.1b --few_shot_num 50 --dataset_name law
