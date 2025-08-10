from beir import LoggingHandler
import logging
import os

data_dir = os.path.join("retrieve", "datasets")

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

out_dir = os.path.join(data_dir, "raw", "beir")
os.makedirs(out_dir, exist_ok=True)
# datasets = ["law"]#, "fiqa"]
# for dataset in datasets:
#     url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
#     data_path = util.download_and_unzip(url, out_dir)
#     print(f"Dataset {dataset} download successfully ...")

# TODO: datasets/raw/beir/law 폴더 만들고, corpus_filtered.jsonl, corpus.jsonl, queries.jsonl, qrels/train.csv, test.csv 넣어두기