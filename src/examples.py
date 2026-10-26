"""Dump a few qualitative examples of batch-size divergence (B=1 vs B=64) that flip correctness.

Output: results/examples.md
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    pairs = pd.read_csv(f"{ROOT}/results/pairs.csv")
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/datasets/processed/items.jsonl")}
    flips = pairs[(pairs.comp == "b1 vs b64") & (pairs.correct_ref != pairs.correct_other)]
    out = ["# Divergence examples (B=1 vs B=64, greedy decoding)\n"]
    shown = 0
    for model in ["q15", "l1b", "q3", "q05"]:
        sub = flips[flips.model == model].sort_values("div_pos")
        for task in ["gsm8k", "mmlu_cot", "agnews"]:
            s = sub[sub.task == task]
            if s.empty:
                continue
            r = s.iloc[len(s) // 2]  # a typical (median-position) divergence, not the most dramatic
            g = {c: json.loads(l) for c in ["b1", "b64"]
                 for l in open(f"{ROOT}/results/gen/{model}/{c}.jsonl") if json.loads(l)["id"] == r.id}
            d = int(r.div_pos)
            a, b = g["b1"], g["b64"]
            prefix_chars = len(a["text"]) if d >= len(a["token_ids"]) else None
            out.append(f"## {model} / {task} / {r.id}\n")
            out.append(f"Gold: `{items[r.id]['gold']}`. First divergence at generated token {d} "
                       f"(B=1 top-1 vs top-2 log-prob margin there: {r.margin_at_div}). "
                       f"B=1 correct: {bool(r.correct_ref)}, B=64 correct: {bool(r.correct_other)}.\n")
            for name, x in [("B=1", a), ("B=64", b)]:
                txt = x["text"].replace("\n", " ")
                out.append(f"**{name}** ({len(x['token_ids'])} tokens): {txt[:900]}{'…' if len(txt) > 900 else ''}\n")
            shown += 1
    open(f"{ROOT}/results/examples.md", "w").write("\n".join(out))
    print(f"wrote {shown} examples")


if __name__ == "__main__":
    main()
