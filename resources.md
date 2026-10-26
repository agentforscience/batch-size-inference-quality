# Resources Used

The resource-finding stage produced no `papers/`, `datasets/` or `code/` in this workspace, so all
resources came from the host's local Hugging Face cache (`/data/transformers/hub`).

- **Datasets** (cached parquet): openai/gsm8k (test), cais/mmlu (all/test), fancyzhx/ag_news (test),
  stanfordnlp/sst2 (validation). Sampling is in `src/prepare_data.py` (seed 42); the result is
  `datasets/processed/items.jsonl`.
- **Models** (cached safetensors): Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct, Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct-AWQ,
  unsloth/Llama-3.2-{1B,3B}-Instruct.
- **Engine:** vLLM 0.32.0 (V1, in-process engine), torch 2.13 + CUDA 13.0, RTX A6000.
- **Literature (background knowledge, not downloaded):** Thinking Machines Lab 2025 (batch
  invariance); Yuan et al. 2025 (FP32 reproducibility); Atil et al. 2024 (LLM stability). See
  REPORT.md §3.
