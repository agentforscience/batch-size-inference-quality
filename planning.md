# Planning: Batch Size Effects on LLM Greedy Inference

## Motivation & Novelty Assessment

### Why This Research Matters
Evaluation pipelines and production serving almost always batch requests, and practitioners routinely
assume that `temperature=0` makes outputs a deterministic function of the prompt. If the batch a
request lands in changes its greedy output, then benchmark numbers, regression tests, cached-answer
systems, and "reproducible" evaluations become irreproducible in ways that are invisible to the user.
How much this matters (how often, for which tasks, by how much accuracy) decides whether
batch-invariant kernels are worth their throughput cost.

### Gap in Existing Work
No resource-finder output was available in this workspace (no `papers/`, `datasets/`, `code/`), so this
review is from background knowledge. The problem is widely known: Thinking Machines Lab's "Defeating
Nondeterminism in LLM Inference" (He et al., 2025) traces it to batch-dependent reduction orders in
matmul, RMSNorm and attention kernels and proposes batch-invariant kernels. Yuan et al. (2025),
"Give Me FP32 or Give Me Death?", report accuracy swings of up to several points for reasoning models
across GPU count, batch size and dtype. Atil et al. (2024) document non-determinism of
"deterministic" API settings. What is missing is a **controlled, per-prompt** decomposition on open
models that (a) separates batch size from batch *composition* and from run-to-run noise, (b) separates
task type from output length (both vary together in most studies), (c) explains *where* in a sequence a
divergence starts (logit near-ties), and (d) compares model sizes on a single numeric format.

### Our Novel Contribution
We run byte-identical prompts through a single vLLM engine per model under static batch sizes
B ∈ {1, 4, 16, 64}, plus a same-B reshuffle and same-schedule repeats. We measure exact-match rate,
first-divergence position, the reference model's top-1/top-2 logit margin at the divergence token,
and correctness flips, on 5 tasks that cross {math, non-math reasoning, classification} with
{long CoT, short answer}, and on 6 models (Qwen2.5 0.5B/1.5B/3B/7B, Llama-3.2 1B/3B) plus an
fp32 control.

### Experiment Justification
- **Exp 1 (batch-size sweep)**: directly tests the >95%-exact-match claim and the dose-response
  with B.
- **Exp 2 (controls: b64 repeat, b1 repeat, b64 reshuffle)**: without these we can't tell batch-size
  effects from general run-to-run nondeterminism or from which prompts share a batch.
- **Exp 3 (task × length design)**: MMLU appears both with CoT (long) and as a direct letter answer
  (short), so the "math vs classification" claim can be separated from output length. Survival
  and hazard analysis of the first-divergence position normalises for length.
- **Exp 4 (accuracy impact)**: tests "<1% mean accuracy difference but individual flips", with paired
  tests.
- **Exp 5 (model size + fp32 control)**: tests "smaller models diverge more". The fp32 run of the
  0.5B model tests the floating-point explanation causally: higher precision should almost remove
  divergence.

## Research Question
Under greedy decoding, does the static batch size used at inference change a model's outputs? How
often, where in the sequence, for which tasks and model sizes, and with what effect on accuracy?

## Hypothesis Decomposition
- H1: ≥95% of prompts give token-identical outputs across batch sizes (only 3–5% diverge).
- H2: divergence probability rises with output length.
- H3: divergence is more common for math/reasoning than for classification, *beyond* the length effect.
- H4: mean accuracy differs by <1 pp across batch sizes, but individual items flip correct↔incorrect.
- H5: smaller models diverge more than larger ones.
- Mechanism (H0-mech): divergences start at tokens where the reference top-1/top-2 margin is ~0, and
  fp32 inference sharply reduces them.

## Proposed Methodology

### Approach
vLLM 0.32 (V1 engine, eager mode, chunked prefill, prefix caching disabled, max_num_seqs=64), with
one engine per model. Batch size is controlled by submitting prompts in static chunks of B
(`llm.generate(chunk)`). Prompts are in a fixed, task-mixed random order (as a serving queue would see
them). All sampling is greedy (temperature 0), with top-2 logprobs recorded per token.

### Experimental Steps
1. Build 1,700 items: GSM8K 400, MMLU-CoT 300, MMLU-direct 400, AG News 300, SST-2 300.
2. For each model, run conditions b64, b64_rep, b64_shuf, b16, b4, b1, b1_rep (b1_rep on every 5th item).
3. Score accuracy per task (number / letter / label extraction).
4. Pairwise exact-match, first divergence index, margin at divergence, answer-flip analysis.

### Baselines
Reference = b1 (sequential). Noise floors = b1 vs b1_rep and b64 vs b64_rep.

### Evaluation Metrics
Exact token-sequence match rate, first-divergence token index, final-answer agreement, task accuracy,
per-token divergence hazard, and the top-1/top-2 logprob margin.

### Statistical Analysis Plan
Wilson 95% CIs for proportions. McNemar exact tests for paired accuracy (b1 vs bX). Cochran's Q
across all B. Logistic regression of divergence on log(output length) + task type + model. Bootstrap
CIs over items for accuracy differences. α = 0.05 with Holm correction across the model × B family.

## Expected Outcomes
H1 is supported if the exact-match rate is ≥95%. The smoke test (0.5B, 61 items, b64 vs b1) already
showed only ~67% exact match, so H1 is likely refuted for long CoT outputs.

## Timeline and Milestones
Setup + smoke test: 30 min (done). Generation: ~2–4 h, depending on shared-GPU availability (queue).
Analysis: 1 h. Report: 45 min.

## Potential Challenges
- Shared GPUs (≤5 GB free at times): use a queue with in-process memory holding; the 7B model may not
  get a slice, in which case b1 runs on a subset (B1_SUB) or is dropped and documented.
- b1 is slow (sequential decode): ~1 h for small models.
- Other tenants' load does not change our kernels' reduction order, but our static chunks mean the
  batch composition is controlled.

## Success Criteria
All small models complete all conditions; at least one ≥3B model completes; every hypothesis component
gets a quantified verdict with CIs.

## Deviations During Execution (added after the runs)
- The Qwen2.5-7B bf16 run never obtained a ≥19 GB GPU slice on the shared machine and was dropped.
  Replacements: Qwen2.5 {0.5,1.5,3}B-AWQ (fp16 activations) and a Qwen2.5-0.5B **fp16** run, added to
  separate activation precision from quantisation after AWQ showed much lower divergence.
- Qwen2.5-3B and Llama-3.2-3B (bf16): b4 dropped and b1 run on the 340-item subset (B1_SUB=1) for
  time reasons. b1_rep was skipped for Qwen2.5-3B and Qwen2.5-0.5B-AWQ.
- Holm family = 28 tests (model × B vs B=1).
