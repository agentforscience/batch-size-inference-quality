"""Score generations and analyse batch-size-induced divergence.

Reads results/gen/<model>/<cond>.jsonl (from generate.py) and writes
  results/pairs.csv          one row per (model, comparison, item): exact match, first divergence, flips
  results/accuracy.csv       accuracy per model x condition x task
  results/summary_*.csv      aggregate tables used in REPORT.md
  results/stats.json         hypothesis tests
  figures/*.png
"""
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = f"{ROOT}/results/gen"
FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)

MODELS = {  # tag -> (label, params in B, dtype)
    "q05": ("Qwen2.5-0.5B", 0.5, "bf16"), "q15": ("Qwen2.5-1.5B", 1.5, "bf16"),
    "q3": ("Qwen2.5-3B", 3.1, "bf16"), "q7": ("Qwen2.5-7B", 7.6, "bf16"),
    "l1b": ("Llama-3.2-1B", 1.2, "bf16"), "l3b": ("Llama-3.2-3B", 3.2, "bf16"),
    "q05f32": ("Qwen2.5-0.5B fp32", 0.5, "fp32"), "q3awq": ("Qwen2.5-3B AWQ", 3.1, "awq"),
    "q7awq": ("Qwen2.5-7B AWQ", 7.6, "awq"), "q15awq": ("Qwen2.5-1.5B AWQ", 1.5, "awq"),
    "q05awq": ("Qwen2.5-0.5B AWQ", 0.5, "awq"), "q05f16": ("Qwen2.5-0.5B fp16", 0.5, "fp16"),
}
TASKS = ["gsm8k", "mmlu_cot", "mmlu_direct", "agnews", "sst2"]
LONG = {"gsm8k", "mmlu_cot"}
# (comparison name, condition a, condition b); a is the reference whose margins are used
COMPS = [("b1 vs b4", "b1", "b4"), ("b1 vs b16", "b1", "b16"), ("b1 vs b64", "b1", "b64"),
         ("b1 vs b1_rep", "b1", "b1_rep"), ("b64 vs b64_rep", "b64", "b64_rep"),
         ("b64 vs b64_shuf", "b64", "b64_shuf"), ("b4 vs b16", "b4", "b16"), ("b16 vs b64", "b16", "b64")]
BS = {"b1": 1, "b4": 4, "b16": 16, "b64": 64}


# ---------------------------------------------------------------- answer extraction / scoring
def extract(task, text):
    t = text.strip()
    if task == "gsm8k":
        m = re.findall(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", t, flags=re.I)
        if not m:
            m = re.findall(r"-?[\d,]*\.?\d+", t)
        if not m:
            return None
        try:
            v = float(m[-1].replace(",", ""))
            return str(int(v)) if v == int(v) else str(v)
        except ValueError:
            return None
    if task == "mmlu_cot":
        m = re.findall(r"answer is[:\s]*\(?\**([ABCD])\b", t, flags=re.I)
        if m:
            return m[-1].upper()
        m = re.findall(r"\b([ABCD])[\.\)]", t)
        return m[-1] if m else None
    if task == "mmlu_direct":
        m = re.match(r"\W*\(?([ABCD])\b", t)
        return m.group(1) if m else None
    if task == "agnews":
        low = t.lower()
        for key, lab in [("sci", "Sci/Tech"), ("tech", "Sci/Tech"), ("science", "Sci/Tech"),
                         ("sport", "Sports"), ("business", "Business"), ("world", "World")]:
            if low.startswith(key) or re.match(rf"\W*{key}", low):
                return lab
        return None
    if task == "sst2":
        low = t.lower()
        return "positive" if "positive" in low else ("negative" if "negative" in low else None)


def correct(task, pred, gold):
    if pred is None:
        return False
    if task == "gsm8k":
        try:
            return abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            return False
    return pred == gold


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ---------------------------------------------------------------- load
def load():
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/datasets/processed/items.jsonl")}
    data = {}
    for tag in sorted(os.listdir(GEN)):
        conds = {}
        for f in os.listdir(f"{GEN}/{tag}"):
            if f.endswith(".jsonl") and f != "timing.jsonl":
                recs = {}
                for l in open(f"{GEN}/{tag}/{f}"):
                    r = json.loads(l)
                    it = items[r["id"]]
                    r["pred"] = extract(it["task"], r["text"])
                    r["correct"] = correct(it["task"], r["pred"], it["gold"])
                    recs[r["id"]] = r
                conds[f[:-6]] = recs
        if conds:
            data[tag] = conds
    return items, data


def build_pairs(items, data):
    rows = []
    for tag, conds in data.items():
        for name, ca, cb in COMPS:
            if ca not in conds or cb not in conds:
                continue
            A, Bc = conds[ca], conds[cb]
            for iid in sorted(A.keys() & Bc.keys()):  # sorted: set order varies with PYTHONHASHSEED
                a, b = A[iid], Bc[iid]
                d = first_div(a["token_ids"], b["token_ids"])
                ma = a["margin"][d] if d is not None and d < len(a["margin"]) else np.nan
                rows.append({
                    "model": tag, "comp": name, "id": iid, "task": a["task"],
                    "len_ref": len(a["token_ids"]), "len_other": len(b["token_ids"]),
                    "exact": d is None, "div_pos": d, "margin_at_div": ma,
                    "rel_div_pos": d / len(a["token_ids"]) if d is not None and len(a["token_ids"]) else np.nan,
                    "answer_same": a["pred"] == b["pred"], "correct_ref": a["correct"], "correct_other": b["correct"],
                    "min_margin_ref": float(np.nanmin([m for m in a["margin"] if m is not None]))
                    if any(m is not None for m in a["margin"]) else np.nan,
                    "n_ties_ref": int(sum(1 for m in a["margin"] if m == 0)),
                    "finish_ref": a["finish"], "finish_other": b["finish"],
                })
    return pd.DataFrame(rows)


def build_acc(items, data):
    rows = []
    for tag, conds in data.items():
        for c, recs in conds.items():
            for iid, r in recs.items():
                rows.append({"model": tag, "cond": c, "id": iid, "task": r["task"], "correct": r["correct"],
                             "parsed": r["pred"] is not None, "len": len(r["token_ids"])})
    return pd.DataFrame(rows)


def prop_table(df, by):
    g = df.groupby(by)
    out = g.agg(n=("exact", "size"), exact=("exact", "mean"), answer_same=("answer_same", "mean")).reset_index()
    out["diverge_pct"] = 100 * (1 - out["exact"])
    ci = [wilson(int(round((1 - e) * n)), n) for e, n in zip(out["exact"], out["n"])]
    out["div_lo"] = [100 * c[0] for c in ci]
    out["div_hi"] = [100 * c[1] for c in ci]
    out["answer_change_pct"] = 100 * (1 - out["answer_same"])
    return out


def main():
    items, data = load()
    pairs = build_pairs(items, data)
    acc = build_acc(items, data)
    pairs.to_csv(f"{ROOT}/results/pairs.csv", index=False)
    acc.to_csv(f"{ROOT}/results/accuracy_items.csv", index=False)
    st = {}
    lab = lambda t: MODELS.get(t, (t,))[0]  # noqa: E731

    # ---- 1. divergence by model x comparison (all tasks / long / short)
    pairs["kind"] = np.where(pairs["task"].isin(LONG), "long (CoT)", "short (label)")
    t1 = prop_table(pairs, ["model", "comp"])
    t1k = prop_table(pairs, ["model", "comp", "kind"])
    t1t = prop_table(pairs, ["model", "comp", "task"])
    t1.to_csv(f"{ROOT}/results/summary_divergence.csv", index=False)
    t1k.to_csv(f"{ROOT}/results/summary_divergence_kind.csv", index=False)
    t1t.to_csv(f"{ROOT}/results/summary_divergence_task.csv", index=False)

    # ---- 2. accuracy per model x cond x task, and McNemar b1 vs bX
    acc_t = acc.groupby(["model", "cond", "task"]).correct.mean().unstack("task") * 100
    acc_t["macro"] = acc_t[[t for t in TASKS if t in acc_t]].mean(axis=1)
    acc_t["micro"] = acc.groupby(["model", "cond"]).correct.mean() * 100
    acc_t["n_items"] = acc.groupby(["model", "cond"]).size()  # <1700 = subset condition (b1_rep, B1_SUB)
    acc_t.round(2).to_csv(f"{ROOT}/results/summary_accuracy.csv")
    mc = []
    for tag in data:
        for cb in ["b4", "b16", "b64", "b64_rep", "b64_shuf"]:
            sub = pairs[(pairs.model == tag) & (pairs.comp.isin([f"b1 vs {cb}"]))]
            if cb in ("b64_rep", "b64_shuf"):
                sub = pairs[(pairs.model == tag) & (pairs.comp == f"b64 vs {cb}")]
            if sub.empty:
                continue
            for scope, s in [("all", sub)] + [(t, sub[sub.task == t]) for t in TASKS]:
                n01 = int(((s.correct_ref) & (~s.correct_other)).sum())  # ref right, other wrong
                n10 = int(((~s.correct_ref) & (s.correct_other)).sum())
                p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
                # paired bootstrap CI for accuracy difference (other - ref)
                diff = s.correct_other.astype(int).values - s.correct_ref.astype(int).values
                rng = np.random.default_rng(0)
                boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)] if len(diff) else [0]
                mc.append({"model": tag, "comparison": sub.comp.iloc[0], "scope": scope, "n": len(s),
                           "acc_ref": 100 * s.correct_ref.mean(), "acc_other": 100 * s.correct_other.mean(),
                           "delta_pp": 100 * diff.mean() if len(diff) else 0,
                           "ci_lo": 100 * np.percentile(boots, 2.5), "ci_hi": 100 * np.percentile(boots, 97.5),
                           "flips_right_to_wrong": n01, "flips_wrong_to_right": n10,
                           "flip_pct": 100 * (n01 + n10) / max(len(s), 1), "mcnemar_p": p})
    mc = pd.DataFrame(mc)
    # Holm correction over the main family (scope=all, b1 vs b4/b16/b64)
    fam = mc[(mc.scope == "all") & mc.comparison.str.startswith("b1 vs")].index
    if len(fam):
        ps = mc.loc[fam, "mcnemar_p"].values
        order = np.argsort(ps)
        adj = np.empty_like(ps)
        running = 0
        for rank, i in enumerate(order):
            running = max(running, min(1, ps[i] * (len(ps) - rank)))
            adj[i] = running
        mc.loc[fam, "p_holm"] = adj
    mc.to_csv(f"{ROOT}/results/summary_accuracy_flips.csv", index=False)

    # ---- 3. length / task analysis: logistic regression on long tasks, b1 vs b64
    import statsmodels.formula.api as smf
    MAIN = os.environ.get("MAIN_COMP", "b1 vs b64")
    main_c = pairs[pairs.comp == MAIN].copy()
    main_c["div"] = (~main_c.exact).astype(int)
    main_c["loglen"] = np.log(main_c.len_ref.clip(lower=1))
    try:
        lr = smf.logit("div ~ loglen + C(task, Treatment('mmlu_cot')) + C(model)",
                       data=main_c[main_c.task.isin(LONG)]).fit(disp=0)
        st["logit_long_tasks"] = {k: {"coef": float(lr.params[k]), "p": float(lr.pvalues[k])} for k in lr.params.index}
        lr2 = smf.logit("div ~ loglen + C(kind) + C(model)", data=main_c).fit(disp=0)
        st["logit_all_tasks"] = {k: {"coef": float(lr2.params[k]), "p": float(lr2.pvalues[k])} for k in lr2.params.index}
    except Exception as e:  # noqa: BLE001
        st["logit_error"] = str(e)

    # per-token hazard: divergences / tokens generated identically before divergence (or to the end)
    def hazard(df):
        at_risk = np.where(df.exact, df[["len_ref", "len_other"]].min(axis=1), df.div_pos.fillna(0) + 1).sum()
        return 1000 * (~df.exact).sum() / max(at_risk, 1)
    hz = pd.DataFrame([{"model": m, "task": t, "div_per_1k_tokens": hazard(g)}
                       for (m, t), g in main_c.groupby(["model", "task"])], columns=["model", "task", "div_per_1k_tokens"])
    hz.to_csv(f"{ROOT}/results/summary_hazard.csv", index=False)

    # where divergences start: first generated token (decided by the prefill pass) vs later decode steps,
    # and how often a divergence changes the extracted answer
    dp = pairs[~pairs.exact].copy()
    dp["at_first_token"] = dp.div_pos == 0
    dpos = dp.groupby(["model", "comp", "kind"]).agg(
        n_div=("exact", "size"), pct_at_first_token=("at_first_token", "mean"),
        median_div_pos=("div_pos", "median"), median_rel_div_pos=("rel_div_pos", "median"),
        pct_answer_changed_given_div=("answer_same", lambda s: 100 * (1 - s.mean()))).reset_index()
    dpos["pct_at_first_token"] *= 100
    dpos.to_csv(f"{ROOT}/results/summary_divpos.csv", index=False)

    # ---- 4. mechanism: margin at divergence vs all reference tokens
    mech = []
    for tag, conds in data.items():
        ref = MAIN.split(" vs ")[0]
        if ref not in conds:
            continue
        allm = np.array([m for r in conds[ref].values() for m in r["margin"] if m is not None])
        dv = main_c[(main_c.model == tag) & (~main_c.exact)].margin_at_div.dropna().values
        mech.append({"model": tag, "frac_tokens_tie": float((allm == 0).mean()),
                     "frac_tokens_margin_le_0.125": float((allm <= 0.125 + 1e-9).mean()),
                     "n_div": len(dv), "frac_div_at_tie": float((dv == 0).mean()) if len(dv) else np.nan,
                     "frac_div_margin_le_0.125": float((dv <= 0.125 + 1e-9).mean()) if len(dv) else np.nan,
                     "median_div_margin": float(np.median(dv)) if len(dv) else np.nan,
                     "max_div_margin": float(dv.max()) if len(dv) else np.nan})
    mech = pd.DataFrame(mech)
    mech.to_csv(f"{ROOT}/results/summary_mechanism.csv", index=False)

    # ---- 5. size: Spearman of divergence vs params (bf16 models, b1 vs b64, long tasks)
    sz = prop_table(main_c[main_c.task.isin(LONG)], ["model"])
    sz = sz[sz["model"].map(lambda t: MODELS.get(t, ("", 0, ""))[2] == "bf16").astype(bool)] if len(sz) else sz
    sz["params"] = sz["model"].map(lambda t: MODELS[t][1]) if len(sz) else []
    if len(sz) >= 3:
        rho, p = stats.spearmanr(sz.params, sz.diverge_pct)
        st["size_spearman_long"] = {"rho": float(rho), "p": float(p), "n_models": len(sz)}
    # tie-propensity vs size
    st["mechanism"] = mech.to_dict(orient="records")
    json.dump(st, open(f"{ROOT}/results/stats.json", "w"), indent=1, default=float)

    # ---------------------------------------------------------------- figures
    order = [m for m in MODELS if m in data]
    colors = dict(zip(order, plt.cm.tab10.colors))

    # F1: divergence % vs batch size (vs b1) per model, long and short
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, kind in zip(axs, ["long (CoT)", "short (label)"]):
        for m in order:
            s = t1k[(t1k.model == m) & (t1k.kind == kind) & t1k.comp.isin(["b1 vs b4", "b1 vs b16", "b1 vs b64"])]
            if s.empty:
                continue
            x = [BS[c.split(" vs ")[1]] for c in s.comp]
            o = np.argsort(x)
            ax.errorbar(np.array(x)[o], s.diverge_pct.values[o],
                        yerr=[s.diverge_pct.values[o] - s.div_lo.values[o], s.div_hi.values[o] - s.diverge_pct.values[o]],
                        marker="o", capsize=3, label=lab(m), color=colors[m])
            fl = t1k[(t1k.model == m) & (t1k.kind == kind) & (t1k.comp == "b1 vs b1_rep")]
            if not fl.empty:
                ax.scatter([1], fl.diverge_pct, marker="x", color=colors[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(["1 (rep)", "4", "16", "64"])
        ax.set_xlabel("batch size B (compared with B=1)"); ax.set_ylabel("% prompts with non-identical output")
        ax.set_title(f"{kind} tasks"); ax.grid(alpha=.3)
    axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/divergence_vs_batch.png", dpi=150); plt.close()

    # F2: divergence by task (b1 vs b64) per model
    s = t1t[t1t.comp == "b1 vs b64"]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / max(len(order), 1)
    for k, m in enumerate(order):
        r = s[s.model == m].set_index("task").reindex(TASKS)
        ax.bar(np.arange(len(TASKS)) + k * w, r.diverge_pct, w, color=colors[m], label=lab(m),
               yerr=[r.diverge_pct - r.div_lo, r.div_hi - r.diverge_pct], capsize=2)
    ax.set_xticks(np.arange(len(TASKS)) + w * (len(order) - 1) / 2); ax.set_xticklabels(TASKS)
    ax.set_ylabel("% prompts diverging (B=1 vs B=64)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/divergence_by_task.png", dpi=150); plt.close()

    # F3: survival of identical prefix (long tasks, b1 vs b64): P(still identical after t tokens)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in order:
        s = main_c[(main_c.model == m) & main_c.task.isin(LONG)]
        if s.empty:
            continue
        T = np.arange(0, 513, 4)
        dpos = s.div_pos.fillna(1e9).values
        surv = [(dpos > t).mean() for t in T]
        ax.plot(T, surv, label=lab(m), color=colors[m])
    ax.set_xlabel("generated token position t"); ax.set_ylabel("P(outputs identical through t)")
    ax.set_title("Long CoT tasks, B=1 vs B=64"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/survival_identical_prefix.png", dpi=150); plt.close()

    # F4: divergence rate vs output length bins (long tasks)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = [0, 100, 150, 200, 250, 300, 400, 513]
    for m in order:
        s = main_c[(main_c.model == m) & main_c.task.isin(LONG)].copy()
        if s.empty:
            continue
        s["bin"] = pd.cut(s.len_ref, bins)
        g = s.groupby("bin", observed=True).exact.agg(["mean", "size"])
        g = g[g["size"] >= 10]
        ax.plot([b.mid for b in g.index], 100 * (1 - g["mean"]), marker="o", label=lab(m), color=colors[m])
    ax.set_xlabel("reference (B=1) output length, tokens"); ax.set_ylabel("% diverging (B=1 vs B=64)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/divergence_vs_length.png", dpi=150); plt.close()

    # F5: margin histogram at divergence vs all tokens
    fig, ax = plt.subplots(figsize=(6.5, 4))
    edges = np.array([-0.01, 0.01, 0.13, 0.26, 0.51, 1.01, 2.01, 100])
    names = ["0 (tie)", "≤0.125", "≤0.25", "≤0.5", "≤1", "≤2", ">2"]
    allm = np.array([m for t in order if t in data and "b1" in data[t] and MODELS[t][2] == "bf16"
                     for r in data[t]["b1"].values() for m in r["margin"] if m is not None])
    dv = main_c[(~main_c.exact) & main_c.model.map(lambda t: MODELS.get(t, ("", 0, ""))[2] == "bf16")].margin_at_div.dropna().values
    h1 = np.histogram(allm, edges)[0] / max(len(allm), 1)
    h2 = np.histogram(dv, edges)[0] / max(len(dv), 1)
    x = np.arange(len(names))
    ax.bar(x - 0.2, 100 * h1, 0.4, label=f"all B=1 tokens (n={len(allm):,})", color="#999")
    ax.bar(x + 0.2, 100 * h2, 0.4, label=f"first-divergence tokens (n={len(dv):,})", color="#c44")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_xlabel("top-1 minus top-2 log-prob in B=1 run"); ax.set_ylabel("% of tokens"); ax.legend(fontsize=8)
    ax.set_yscale("log"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/margin_at_divergence.png", dpi=150); plt.close()

    # F6: accuracy deltas (bX - b1) with bootstrap CI, scope=all
    s = mc[(mc.scope == "all")]
    fig, ax = plt.subplots(figsize=(8, 4))
    comps = ["b1 vs b4", "b1 vs b16", "b1 vs b64", "b64 vs b64_rep", "b64 vs b64_shuf"]
    for k, m in enumerate(order):
        r = s[s.model == m].set_index("comparison").reindex(comps)
        ax.errorbar(np.arange(len(comps)) + (k - len(order) / 2) * 0.08, r.delta_pp,
                    yerr=[r.delta_pp - r.ci_lo, r.ci_hi - r.delta_pp], fmt="o", capsize=2, color=colors[m], label=lab(m))
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(comps))); ax.set_xticklabels(comps, fontsize=8)
    ax.set_ylabel("accuracy change (pp), paired items\n(3B bf16 b1 comparisons: 340-item subset)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/accuracy_delta.png", dpi=150); plt.close()

    # ---- compact markdown table for the report: one row per model
    rows = []
    for m in order:
        r = {"model": lab(m), "dtype": MODELS[m][2]}
        for comp in ["b1 vs b4", "b1 vs b16", "b1 vs b64", "b64 vs b64_shuf", "b64 vs b64_rep", "b1 vs b1_rep"]:
            for kind in ["long (CoT)", "short (label)"]:
                s = t1k[(t1k.model == m) & (t1k.comp == comp) & (t1k.kind == kind)]
                r[f"{comp} ({kind[:5]})"] = (f"{s.diverge_pct.iloc[0]:.1f} (n={int(s.n.iloc[0])})" if not s.empty else "–")
        mm = mech[mech.model == m]
        r["% B=1 tokens tied"] = f"{100 * mm.frac_tokens_tie.iloc[0]:.2f}" if not mm.empty else "–"
        f = mc[(mc.model == m) & (mc.scope == "all") & (mc.comparison == "b1 vs b64")]
        r["Δacc b64−b1 (pp)"] = (f"{f.delta_pp.iloc[0]:+.2f} [{f.ci_lo.iloc[0]:+.2f},{f.ci_hi.iloc[0]:+.2f}]"
                                if not f.empty else "–")
        r["flip % (b1 vs b64)"] = f"{f.flip_pct.iloc[0]:.1f}" if not f.empty else "–"
        rows.append(r)
    pd.DataFrame(rows).to_markdown(f"{ROOT}/results/main_table.md", index=False)

    # ---- console summary
    pd.set_option("display.width", 200)
    print(t1.pivot(index="model", columns="comp", values="diverge_pct").round(1))
    print(t1k[t1k.comp == "b1 vs b64"][["model", "kind", "n", "diverge_pct", "div_lo", "div_hi", "answer_change_pct"]].round(1))
    print(acc_t.round(1))
    print(mc[mc.scope == "all"].round(3).to_string())
    print(mech.round(4).to_string())
    print(hz.pivot(index="model", columns="task", values="div_per_1k_tokens").round(2))
    print(json.dumps({k: v for k, v in st.items() if k != "mechanism"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
