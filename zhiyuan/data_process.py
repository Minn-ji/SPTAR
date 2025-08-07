from tqdm import tqdm
import copy
import os
from os.path import join
import pandas as pd
import json
import argparse
import random

random.seed(1359)

cwd = os.getcwd()
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
beir_dir = join(raw_dir, "beir")
weak_dir = join(data_dir, "weak")
xuyang_data_dir = join(cwd, "xuyang", "data")

def read_json(json_path):
    data = []
    for line in open(json_path, 'r'):
        data.append(json.loads(line))
    return data

def read_weak_json(json_path):
    # read weak query to dict for easily remove unqualified queries
    data = {}
    for line in open(json_path, 'r'):
        weak_q = json.loads(line)
        data[weak_q["_id"]] = weak_q
    return data


def filter_unlabeled_corpus(data_path):
    '''이미 쿼리와 연결된 문서를 제외한
    순수한 '레이블 없는 문서'들만 (C_unlabeled)을 corpus_filtered.jsonl 파일로 저장하는 역할'''
    filtered_corpus = []
    filtered_corpus_path = join(data_path, "corpus_filtered.jsonl")
    if os.path.exists(filtered_corpus_path): # 기존에 해당 파일이 있으면 새로 생성
        os.remove(filtered_corpus_path)
        print(f"old {filtered_corpus_path} is deleted ...")

    # 각 데이터셋 다운로드시 아래의 파일이 존재함
    corpus_path = join(data_path, "corpus.jsonl")  # 전체 문서 (C)
    train_path = join(data_path, "qrels", "train.tsv")  # 훈련 쿼리-문서 관련성 데이터
    dev_path = join(data_path, "qrels", "dev.tsv")    # 개발 쿼리-문서 관련성 데이터
    test_path = join(data_path, "qrels", "test.tsv")  # 테스트 쿼리-문서 관련성 데이터
    # _id, title, text, metadata

    corpus = read_json(corpus_path)  # 전체 데이터 부르고
    labeled_corpus_set = set()

    # 모든 qrels 데이터를 하나로 합침(전체 데이터에서 아래 데이터에 등장했으면 query(정답)가 있는것)
    train_data = pd.read_csv(train_path, sep='\t')
    dev_data = pd.read_csv(dev_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    df = pd.concat([train_data, dev_data, test_data])

    # df query-id, corpus-id, score
    # 합쳐진 데이터프레임에서 'corpus-id' (문서 ID)를 추출하여 labeled_corpus_set에 추가
    # 'corpus-id'가 qrels 파일에 있다는 것은 해당 문서에 쿼리가 연결되어 있다는 의미이므로 '레이블이 있는 문서'로 간주하는 것
    for _, row in df.iterrows():
        labeled_corpus_set.add(str(row["corpus-id"]))
    for doc in tqdm(corpus):
        if doc["_id"] in labeled_corpus_set:
            continue
        else: # 레이블이 없는 문서들의 경우 filtered_corpus에 추가됨
            filtered_corpus.append(doc)
    random.shuffle(filtered_corpus) # 섞고
    print(f"writing {filtered_corpus_path} ...")

    with open(filtered_corpus_path, "w+") as f:
        for doc in filtered_corpus:
            json.dump(doc, f)
            f.write("\n") # 라벨 없는 dataset 완성 => "corpus_filtered.jsonl"에 저장

def sample_corpus(dataset_name, ratio: int = 20, train_num: int = 50, weak_num: str = "100k"):
    """sample positive: negative = ration
    데이터 증강을 위해 약한 쿼리(weak query)를 생성할 문서들을 준비하는 역할
    Args:
        folder_path (_type_): _description_
    """
    reduced_corpus, sample_negative_corpus = [], []
    filtered_corpus_path = join(beir_dir, dataset_name, "corpus_filtered.jsonl") # 라벨 없는 데이터
    # 최종적으로 샘플링된 문서들을 저장할 경로
    reduced_corpus_path = join(beir_dir, dataset_name, f"corpus_{weak_num}_reduced_ratio_{ratio}.jsonl")
    # 약한 쿼리가 생성될 예정인 문서 ID(=100k) 목록
    sampled_corpus_path = join(xuyang_data_dir, f"{dataset_name}_{train_num}", weak_num, f"corpus_filtered_{weak_num}_id.tsv")
    if os.path.exists(reduced_corpus_path):
        os.remove(reduced_corpus_path)
        print(f"old {reduced_corpus_path} is deleted ...")
    filtered_corpus = read_json(filtered_corpus_path)
    corpus_path = join(beir_dir, dataset_name, "corpus.jsonl") # 전체 데이터

    # Soft Prompt 튜닝에 사용될 훈련 데이터 (D_train의 일부)
    # TODO: prompt_tuning_{train_num}.tsv는 따로 구축해야 함!!
    # D_train (원본 MS MARCO 훈련 데이터)에서 train_num (예: 50)개의 고유한 쿼리와 그에 해당하는 문서들을 무작위로 샘플링하여 생성한 파일
    train_path = join(xuyang_data_dir, f"{dataset_name}_{train_num}", f"prompt_tuning_{train_num}.tsv")
    # 쿼리-문서 관련성 데이터
    dev_path = join(beir_dir, dataset_name, "qrels", "dev.tsv")
    test_path = join(beir_dir, dataset_name, "qrels", "test.tsv")
    # _id, title, text, metadata
    corpus = read_json(corpus_path)
    labeled_corpus_set = set()
    train_data = pd.read_csv(train_path, sep='\t')
    dev_data = pd.read_csv(dev_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    sampled_data = pd.read_csv(sampled_corpus_path, sep='\t')
    df = pd.concat([train_data, dev_data, test_data])
    # df query-id, corpus-id, score
    for _, row in df.iterrows():
        labeled_corpus_set.add(str(row["corpus-id"]))
    for _, row in sampled_data.iterrows():
        labeled_corpus_set.add(str(row["_id"]))
    # sample negative from filtered_corpus-weak
    filtered_corpus_remove_weak = [] # 약한 쿼리 생성 대상 문서 + 레이블된 문서 제외한 나머지 레이블 없는 문서들
    for filter_doc in tqdm(filtered_corpus):
        if filter_doc["_id"] in labeled_corpus_set:
            continue
        else:
            filtered_corpus_remove_weak.append(filter_doc)
    for doc in tqdm(corpus):
        if doc["_id"] in labeled_corpus_set:
            reduced_corpus.append(doc) 
    sample_num = len(labeled_corpus_set) * ratio
    if sample_num > len(corpus):
        sample_num = len(corpus)
        print(f"{dataset_name} samples all the corpus")
        with open(reduced_corpus_path, "w+") as f:
            for reduce_doc in corpus:
                json.dump(reduce_doc, f)
                f.write("\n")
        return
    random.seed(3490856385)
    # 부정 문서 샘플링
    sample_negative_corpus = random.sample(filtered_corpus_remove_weak, sample_num)
    assert len(sample_negative_corpus) == sample_num
    reduced_corpus += sample_negative_corpus
    random.shuffle(reduced_corpus)
    print(f"writing {len(reduced_corpus)} documents to {reduced_corpus_path} ...")
    with open(reduced_corpus_path, "w+") as f:
        for reduce_doc in reduced_corpus:
            json.dump(reduce_doc, f)
            f.write("\n")

def load_dl(dir):
    """
    load dl2019 dl2020 queries and qrels
    """
    queries = {}
    year = dir.split("_")[-1]
    with open(join(dir, f"queries_{year}", "raw.tsv")) as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query
    qrels_binary = read_json(join(dir, "qrel_binary.json"))[0]
    qrels = read_json(join(dir, "qrel.json"))[0]
    return queries, qrels, qrels_binary

def merge_queries(queries, queries_19, queries_20):
    """
    merge ms's queries with the queries of DL2019 and DL2020 for quick retrieval
    """
    ms_queries = copy.deepcopy(queries)
    for qid, query in queries_19.items():
        ms_queries["tr19ec"+qid] = query
    for qid, query in queries_20.items():
        ms_queries["tr20ec"+qid] = query
    return ms_queries

def extract_results(ms_results):
    """
    extract the results of MS, DL2019 and DL2020 
    """
    results, results_19, results_20 = {}, {}, {}
    for qid, res in ms_results.items():
        if "tr19ec" in qid:
            results_19[qid[6:]] = res
        elif "tr20ec" in qid:
            results_20[qid[6:]] = res
        elif "trec2019" in qid:
            results_19[qid[8:]] = res
        elif "trec2020" in qid:
            results_20[qid[8:]] = res
        else:
            results[qid] = res
    return results, results_19, results_20

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
    args = parser.parse_args()
    return args

def main():
    datasets = ["msmarco"]#, "fiqa"]
    # W_large: 100,000개의 레이블 없는 문서에 대해 약한 쿼리를 생성한 데이터셋.
    # W_small: W_large에서 5,000개의 문서-쿼리 쌍을 샘플링한 데이터셋.
    # C_unlabeled에서 샘플링한 문서들에 대해 LLM이 Soft Prompt Augmentor를 통해 생성한 '약한 쿼리'를 짝지어 만든 새로운 데이터셋
    weak_nums = ["100k"]#, "5000"]
    for dataset_name in tqdm(datasets):
        for weak_num in weak_nums:
            # folder_path = join(beir_dir, dataset_name)
            # filter_unlabeled_corpus(folder_path)
            sample_corpus(dataset_name, weak_num=weak_num)

if __name__ == "__main__":
    main()

