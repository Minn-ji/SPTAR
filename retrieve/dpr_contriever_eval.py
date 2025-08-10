#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import re
import subprocess
from pathlib import Path

# TODO: 가상환경 Python 고정! 내 가상환경 파이썬위치로 바꿔야 함
VENV_PY = r"C:\xai_6th_adv\.venv\Scripts\python.exe"

def as_list(exp_names):
    if isinstance(exp_names, str):
        # "no_aug,exp2  exp3" 모두 허용
        return [x for x in re.split(r"[,\s]+", exp_names.strip()) if x]
    return list(exp_names or [])

def run(cmd, env=None, cwd=None):
    print("RUN:", " ".join(cmd))
    # 출력이 바로 콘솔에 흐르도록 capture_output=False
    res = subprocess.run(cmd, env=env, cwd=cwd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"[ERROR] command failed with exit code {res.returncode}")

def multirun(args):
    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 프로젝트 루트(필요시 조정: -m 임포트 에러 나면 여기 경로를 레포 루트로 바꿔줘)
    cwd = None  # Path(__file__).resolve().parents[1]

    exp_list = as_list(args.exp_names)
    if not exp_list:
        exp_list = ["default"]

    for exp in exp_list:
        print(f"GPU {args.gpu_id} Training: {args.dataset_name} on {exp}")

        if args.version == "v1":
            train_cmd = [
                VENV_PY, "-m", "retrieve.retriever.dpr_contriever.train.train_sbert",
                "--dataset_name", str(args.dataset_name),
                "--train_num", str(args.train_num),
                "--exp_name", exp
            ]
        else:
            raise SystemExit(f"Unknown version: {args.version}")

        run(train_cmd, env=env, cwd=cwd)

        eval_cmd = [
            VENV_PY, "-m", "retrieve.retriever.dpr_contriever.eval.evaluate_sbert",
            "--dataset_name", str(args.dataset_name),
            "--train_num",   str(args.train_num),
            "--dpr_v",       str(args.version),
            "--exp_name", exp
        ]
        run(eval_cmd, env=env, cwd=cwd)
        print()

def main():
    p = argparse.ArgumentParser(description="Training Starts ...")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--version",     type=str, required=True)
    p.add_argument("--gpu_id",      type=int, default=0)
    p.add_argument("--train_num",   type=int, required=True)
    p.add_argument("--exp_names",   type=str, default="no_aug")
    args = p.parse_args()
    multirun(args)

if __name__ == "__main__":
    main()
# python -m retrieve.dpr_contriever_eval --dataset_name law --version v1 --gpu_id 0 --train_num 100 --exp_names no_aug