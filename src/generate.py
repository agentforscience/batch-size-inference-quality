"""Greedy generation for every item under several batch-size conditions, all in ONE vLLM engine.

Usage: python src/generate.py <hf_model_id> <tag> <gpu_mem_gb> [dtype]

A condition is (name, batch size B, prompt order, item subset). For each condition the item list
is put in the given order and submitted to the engine in static chunks of B prompts
(llm.generate(chunk)), so at most B sequences are ever co-scheduled; B=1 is fully sequential.
Using one engine for all conditions removes engine-initialisation as a confound; prefix caching is
disabled so every condition recomputes every prompt from scratch.

Output: results/gen/<tag>/<cond>.jsonl, one line per item with output text, token ids, and the
per-token top-1 log-prob plus top1-top2 log-prob margin (used to explain where divergences occur).
Existing condition files are skipped, so the script is resumable.
"""
import json
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("HF_HOME", "/data/transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 1234
MAX_NUM_SEQS = 64

# (name, B, order seed, subset rule). Ordered cheap-first so partial runs are still useful.
CONDITIONS = [
    ("b64", 64, 0, "all"),
    ("b64_rep", 64, 0, "all"),     # identical schedule again -> run-to-run noise floor
    ("b64_shuf", 64, 1, "all"),    # same B, different batch-mates
    ("b16", 16, 0, "all"),
    ("b4", 4, 0, "all"),
    ("b1", 1, 0, "all"),           # sequential reference
    ("b1_rep", 1, 0, "sub"),       # determinism check of the reference itself (every 5th item)
]

SYSTEM = {
    "gsm8k": "Solve the math word problem. Think step by step, then finish with a final line of the form "
             "'The answer is N' where N is a number.",
    "mmlu_cot": "Answer the multiple-choice question. Think step by step, then finish with a final line of "
                "the form 'The answer is X' where X is one of A, B, C, D.",
    "mmlu_direct": "Answer the multiple-choice question with only the letter (A, B, C, or D) of the correct "
                   "option. Do not explain.",
    "agnews": "Classify the news article into one of these topics: World, Sports, Business, Sci/Tech. "
              "Reply with only the topic name.",
    "sst2": "Classify the sentiment of the movie review sentence as positive or negative. "
            "Reply with only one word: positive or negative.",
}
MAX_TOKENS = {"gsm8k": 512, "mmlu_cot": 512, "mmlu_direct": 8, "agnews": 8, "sst2": 8}


def messages(it):
    return [{"role": "system", "content": SYSTEM[it["task"]]}, {"role": "user", "content": it["question"]}]


def gpu_total_gb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
                                   "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]])
    return float(out.decode().strip()) / 1024


_HOLD, _PATCHED = [], []


def hold_gpu_memory(mem_gb):
    """Shared-GPU startup-race fix (see neurico notes): wait until a GPU has room, grab the budget with a
    placeholder tensor, and free it only when vLLM takes its memory snapshot (in-process engine)."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    fixed = os.environ.get("CUDA_VISIBLE_DEVICES")
    torch = None
    while True:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                       "--format=csv,noheader,nounits"]).decode().split("\n")
        free_by = {int(a): float(b) / 1024 for a, b in (l.split(",") for l in out if l.strip())}
        if fixed is None:
            gpu, free = max(free_by.items(), key=lambda kv: kv[1])
        else:
            gpu, free = int(fixed), free_by[int(fixed)]
        if free >= mem_gb + 0.3:
            if torch is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                import torch
                fixed = str(gpu)
            try:
                _HOLD.append(torch.empty(int(mem_gb * 2**30), dtype=torch.uint8, device="cuda"))
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
        time.sleep(3)
    print(f"holding {mem_gb} GB on GPU {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    if _PATCHED:
        return
    _PATCHED.append(True)
    import vllm.utils.mem_utils as mu
    import vllm.v1.worker.gpu_worker as gw
    Orig = mu.MemorySnapshot

    class HeldSnapshot(Orig):
        def __post_init__(self):
            if _HOLD:
                _HOLD.clear()
                torch.cuda.empty_cache()
            super().__post_init__()
    gw.MemorySnapshot = HeldSnapshot
    mu.MemorySnapshot = HeldSnapshot


def order_items(items, seed, subset):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)  # task-mixed batches, as in a real serving queue
    if subset == "sub":
        idx = [i for i in idx if i % 5 == 0]
    return idx


def summarize(o):
    c = o.outputs[0]
    lp, margin = [], []
    for tid, d in zip(c.token_ids, c.logprobs or []):
        vals = sorted((x.logprob for x in d.values()), reverse=True)
        lp.append(round(d[tid].logprob, 5))
        margin.append(round(vals[0] - vals[1], 5) if len(vals) > 1 else None)
    return {"text": c.text, "token_ids": list(c.token_ids), "lp": lp, "margin": margin,
            "finish": c.finish_reason, "n_prompt": len(o.prompt_token_ids)}


def main():
    model_id, tag, mem_gb = sys.argv[1], sys.argv[2], float(sys.argv[3])
    dtype = sys.argv[4] if len(sys.argv) > 4 else "auto"
    conds = CONDITIONS
    if os.environ.get("CONDS"):
        conds = [c for c in CONDITIONS if c[0] in os.environ["CONDS"].split(",")]
    out_dir = f"{ROOT}/results/gen/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    todo = [c for c in conds if not os.path.exists(f"{out_dir}/{c[0]}.jsonl")]
    if not todo:
        print(f"[{tag}] all conditions done")
        return
    if os.environ.get("HOLD", "1") == "1":
        hold_gpu_memory(mem_gb)
    from vllm import LLM, SamplingParams

    util = mem_gb / gpu_total_gb()
    llm = None
    for attempt in range(40):
        try:
            llm = LLM(model=model_id, dtype=dtype, gpu_memory_utilization=util, max_model_len=2048,
                      max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=1024, seed=SEED,
                      enforce_eager=True, enable_prefix_caching=False, max_logprobs=2)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] engine init failed ({attempt}): {e}", flush=True)
            time.sleep(20)
            if os.environ.get("HOLD", "1") == "1":
                hold_gpu_memory(mem_gb)
    assert llm is not None, "could not start engine"
    tok = llm.get_tokenizer()

    items = [json.loads(l) for l in open(f"{ROOT}/datasets/processed/items.jsonl")]
    if os.environ.get("LIMIT"):  # smoke tests
        items = items[::len(items) // int(os.environ["LIMIT"])]
    prompts = [tok.apply_chat_template(messages(it), tokenize=False, add_generation_prompt=True) for it in items]
    params = [SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS[it["task"]], logprobs=2) for it in items]
    meta = {"model": model_id, "tag": tag, "dtype": dtype, "mem_gb": mem_gb,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "max_num_seqs": MAX_NUM_SEQS}
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)

    for name, B, seed, subset in todo:
        if name == "b1" and os.environ.get("B1_SUB") == "1":  # large models: sequential reference on a subset
            subset = "sub"
        idx = order_items(items, seed, subset)
        t0 = time.time()
        recs = []
        for s in range(0, len(idx), B):
            chunk = idx[s:s + B]
            outs = llm.generate([prompts[i] for i in chunk], [params[i] for i in chunk], use_tqdm=False)
            for pos, (i, o) in enumerate(zip(chunk, outs)):
                r = summarize(o)
                r.update({"id": items[i]["id"], "task": items[i]["task"], "batch_idx": s // B, "pos": pos})
                recs.append(r)
            if s // B % max(1, 200 // B) == 0:
                print(f"[{tag}] {name}: {s + len(chunk)}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
        tmp = f"{out_dir}/{name}.jsonl.partial"
        with open(tmp, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, f"{out_dir}/{name}.jsonl")
        print(f"[{tag}] {name} done in {time.time() - t0:.0f}s", flush=True)
        with open(f"{out_dir}/timing.jsonl", "a") as f:
            f.write(json.dumps({"cond": name, "B": B, "n": len(idx), "sec": time.time() - t0}) + "\n")


if __name__ == "__main__":
    main()
