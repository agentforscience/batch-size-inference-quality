# Does Batching Change Greedy LLM Outputs?

This project runs 1,700 byte-identical prompts (GSM8K, MMLU with and without chain-of-thought (CoT),
AG News, SST-2) through vLLM with greedy decoding under batch sizes 1/4/16/64, plus two controls: a
same-size reshuffle and exact repeats. It covers 10 configurations: Qwen2.5 0.5–3B and Llama-3.2
1–3B in bf16, plus fp16, fp32 and AWQ variants. Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **Batching changes far more outputs than the hypothesised 3–5%.** In bf16, B=1 vs any B≥4 changed
  the exact output of 20–34% of prompts: 47–76% of long CoT outputs and 1–4% of short label outputs.
- **What matters is batch composition, not size.** B=4 already has the full effect, and reshuffling
  batch-mates at fixed B=64 gives 83–100% of it. Re-running an identical schedule gave 0
  differences in all 18 repeat checks: the engine is deterministic but batch-dependent.
- **Mechanism: bf16 near-ties.** About 45% of first-divergence tokens are *exact* top-2 ties in bf16,
  and ≥91% are within one bf16 rounding step (only ~1% of all tokens are ties). Precision ladder on
  the same weights: bf16 33.5% → fp16 7.3% → fp32 2.8% of prompts diverge.
- **Length, not task, drives it.** Divergence rises steeply with output length (logit coefficient on
  log length = +1.1, p≈1e-78). Once length is controlled, label tasks are no more stable than CoT
  (p=0.58), and GSM8K math is *less* divergent than MMLU-CoT.
- **Accuracy barely moves, but items flip.** |Δacc| ≤ 1.8 pp, and no test is significant after
  Holm correction. Still, 3–5% of items (up to 16% of MMLU-CoT) flip between correct and incorrect.
  There was no monotone model-size effect (ρ=−0.05).

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt        # vllm 0.32, pandas, scipy, statsmodels, matplotlib, tabulate
python src/prepare_data.py                # -> datasets/processed/items.jsonl (needs HF cache in /data/transformers)
python src/generate.py Qwen/Qwen2.5-0.5B-Instruct q05 3.6 bfloat16   # <model> <tag> <GPU GB> <dtype>
#   (see logs/jobs.txt for all model/tag/budget lines; B1_SUB=1 runs B=1 on a 340-item subset)
python src/analyze.py                     # -> results/*.csv, results/stats.json, figures/*.png
python src/examples.py                    # -> results/examples.md
```
Runtime: 0.5–1.5B models take ~1–3 h for all 7 conditions on a shared A6000 slice (B=1 dominates).
3B models take longer.

## Layout
```
src/prepare_data.py   build the 1,700-item eval set from cached HF parquet files
src/generate.py       one vLLM engine per model; static-batch conditions b1,b4,b16,b64,b64_rep,b64_shuf,b1_rep
src/queue.sh, src/b1_subset_watch.sh, src/gpu_holder.py   shared-GPU job launching helpers
src/analyze.py        scoring, divergence/flip statistics, regression, figures
src/examples.py       qualitative divergence examples
results/gen/<model>/<cond>.jsonl   raw outputs (text, token ids, per-token logprob + top-2 margin)
results/summary_*.csv, stats.json, main_table.md, examples.md
figures/              divergence_vs_batch, divergence_by_task, divergence_vs_length,
                      survival_identical_prefix, margin_at_divergence, accuracy_delta
planning.md           pre-registered plan;  REPORT.md  full report
```
