from sentence_transformers import losses, models, SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib
import os
import torch
import logging
import argparse
import sys
from retrieve.weak_data_loader import WeakDataLoader

data_dir = os.path.join("retrieve", "datasets")
raw_dir = os.path.join(data_dir, "raw")
beir_dir = os.path.join(raw_dir, "beir")
soft_prompt_dir = os.path.join("soft_prompt", "data")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="law", type=str)
parser.add_argument('--num_epochs', required=False, default=20, type=int)
parser.add_argument('--train_num', required=False, default=100, type=int)
parser.add_argument('--product', required=False, default="cosine", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
parser.add_argument('--learning_rate', required=False, default=3e-2, type=float)
args = parser.parse_args()

model_name = "facebook/contriever"
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.exp_name, str(args.train_num), "{}-v1-{}".format(model_name, args.dataset_name))
os.makedirs(model_save_path, exist_ok=True)

# Just some code to print debug information to stdout
fh = logging.FileHandler(os.path.join(model_save_path, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[fh, ch])


if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(corpus_file=os.path.join(beir_dir, args.dataset_name, f"corpus_100k_reduced_ratio_20.jsonl"), query_file=os.path.join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=os.path.join(soft_prompt_dir, args.dataset_name, "prompt_tuning_train_text.csv")).load_custom()
else:
    weak_query_file = os.path.join("inference_output", args.dataset_name, f"weak_queries_50_tiny_llama-1.1b_523_prompt_3.jsonl")
    weak_qrels_file = os.path.join("inference_output", args.dataset_name, f"weak_train_50_tiny_llama-1.1b_523_prompt_3.csv")
    corpus, queries, qrels = WeakDataLoader(corpus_file=os.path.join(beir_dir, args.dataset_name, f"corpus_100k_reduced_ratio_20.jsonl"), query_file=os.path.join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=os.path.join(soft_prompt_dir, args.dataset_name, f"prompt_tuning_train_text.csv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()

dev_corpus, dev_queries, dev_qrels = GenericDataLoader(corpus_file=os.path.join(beir_dir, args.dataset_name, f"corpus_100k_reduced_ratio_20.jsonl"), query_file=os.path.join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=os.path.join(beir_dir, args.dataset_name, "qrels", "test.csv")).load_custom()

#### Provide any sentence-transformers or HF model
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

retriever = TrainRetriever(model=model, batch_size=32)

#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

#### Training SBERT with cosine-product
if args.product == "cosine":
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    score_functions = {'cos_sim': util.cos_sim}
#### training SBERT with dot-product
elif args.product == "dot":
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)
    score_functions = {'dot_score': util.dot_score}


corpus_chunk_size=100000
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)


num_epochs = args.num_epochs
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=ir_evaluator,
    epochs=num_epochs,
    output_path=model_save_path,

    # ↓ SBERT 표준 인자들만 사용
    warmup_steps=warmup_steps,
    evaluation_steps=10,   # -1이면 epoch마다 평가
    optimizer_class=torch.optim.AdamW,   # 최신 AdamW 경로
    optimizer_params={"lr": args.learning_rate},  # 학습률 여기서 지정
    scheduler='WarmupLinear',                  # 선형 스케줄
    use_amp=True                         # GPU면 AMP, CPU면 무시
)