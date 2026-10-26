"""Build the evaluation item set from locally cached HF parquet files.

Tasks (chosen to separate *task type* from *output length*):
  gsm8k       math word problems, chain-of-thought (long output, <=512 tokens)
  mmlu_cot    MMLU multiple choice, chain-of-thought (long output, non-math reasoning mix)
  mmlu_direct MMLU multiple choice, answer letter only (short output)
  agnews      AG News 4-way topic classification, label only (short output)
  sst2        SST-2 binary sentiment classification, label only (short output)

Output: datasets/processed/items.jsonl (one JSON object per item).
"""
import json
import os
import random

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB = "/data/transformers/hub"
SEED = 42
N = {"gsm8k": 400, "mmlu_cot": 300, "mmlu_direct": 400, "agnews": 300, "sst2": 300}


def snap(repo, rel):
    base = f"{HUB}/datasets--{repo.replace('/', '--')}/snapshots"
    return os.path.join(base, os.listdir(base)[0], rel)


def mmlu_question(r):
    opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
    return f"{r['question']}\n\n{opts}", "ABCD"[int(r["answer"])]


def main():
    rng = random.Random(SEED)
    items = []

    gsm = pd.read_parquet(snap("openai/gsm8k", "main/test-00000-of-00001.parquet"))
    for i in rng.sample(range(len(gsm)), N["gsm8k"]):
        r = gsm.iloc[i]
        items.append({"id": f"gsm8k-{i}", "task": "gsm8k", "question": r["question"],
                      "gold": r["answer"].split("####")[-1].strip().replace(",", "")})

    mmlu = pd.read_parquet(snap("cais/mmlu", "all/test-00000-of-00001.parquet"))
    idx = rng.sample(range(len(mmlu)), N["mmlu_cot"] + N["mmlu_direct"])  # disjoint question sets
    for k, i in enumerate(idx):
        task = "mmlu_cot" if k < N["mmlu_cot"] else "mmlu_direct"
        q, gold = mmlu_question(mmlu.iloc[i])
        items.append({"id": f"{task}-{i}", "task": task, "question": q, "gold": gold,
                      "subject": mmlu.iloc[i]["subject"]})

    ag = pd.read_parquet(snap("fancyzhx/ag_news", "data/test-00000-of-00001.parquet"))
    names = ["World", "Sports", "Business", "Sci/Tech"]
    for i in rng.sample(range(len(ag)), N["agnews"]):
        r = ag.iloc[i]
        items.append({"id": f"agnews-{i}", "task": "agnews", "question": r["text"], "gold": names[int(r["label"])]})

    sst = pd.read_parquet(snap("stanfordnlp/sst2", "data/validation-00000-of-00001.parquet"))
    for i in rng.sample(range(len(sst)), N["sst2"]):
        r = sst.iloc[i]
        items.append({"id": f"sst2-{i}", "task": "sst2", "question": r["sentence"].strip(),
                      "gold": ["negative", "positive"][int(r["label"])]})

    assert len({it["id"] for it in items}) == len(items), "duplicate ids"
    os.makedirs(f"{ROOT}/datasets/processed", exist_ok=True)
    with open(f"{ROOT}/datasets/processed/items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(pd.Series([it["task"] for it in items]).value_counts().to_string())


if __name__ == "__main__":
    main()
