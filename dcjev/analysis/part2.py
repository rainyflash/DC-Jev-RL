"""Part II aggregate (also used for III-B and III-C): accuracies per condition (seed mean and spread), paired per-problem
differences against a baseline condition with bootstrap intervals, training curves, and the judge's view of each policy.

    python -m dcjev.analysis.part2 --root /content/work/runs/part2 --out experiments/part2_llm/results
    python -m dcjev.analysis.part2 --root .../part3b --root .../part2 --conds rlvr,rlvr_sub,sub_kev,sub_kev_recal,sub_probe \
        --baseline rlvr_sub --out experiments/part3b_labels/results
    python -m dcjev.analysis.part2 --root .../part3c --root .../part2 --conds rlvr,tree_random,tree_kev --step 100 \
        --out experiments/part3c_tree/results

Runs are <root>/<condition>/s<seed>; several roots are combined (a condition and seed found in two roots is an error).
Without --step the final evaluation is compared (final.json, final_completions.jsonl.gz); with --step N the periodic
evaluation at step N (the log's eval record, evals/stepN.jsonl.gz), so runs of different lengths meet at the same step.
Paired differences: for each benchmark, per-problem accuracy is averaged over the seeds of a condition (greedy sets: the
fraction of seeds that solve it; sampled sets: the mean over seeds x samples); the difference to the baseline is averaged
over problems, with a bootstrap over problems (10k resamples) for the 95% interval. Seed-level spread is reported too.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from dcjev.analysis import read_log

CONDS = ["rlvr", "rlvr_kev_add", "rlvr_kev_gate", "kev_only", "rlvr_ar_add"]
SETS = ["gsm8k_test", "math500", "amc23", "aime24", "aime25", "val", "gsm8k500"]
EXTRA = ("kev_correct_auroc", "kev_correct_auroc_within_group", "ar_correct_auroc", "ar_correct_auroc_within_group",
         "fill_auroc_unlabelled", "fill_ece_unlabelled", "fill_mean_unlabelled", "acc_unlabelled", "frozen_kev_auroc_unlabelled",
         "roots_finished", "on_track_auroc_leaves", "on_track_auroc_leaves_within_prompt", "audit_value_all_roots",
         "audit_selection_gain", "audit_best_possible_gain")


def per_problem(path):
    acc = defaultdict(lambda: defaultdict(list))
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l); acc[r["set"]][r["id"]].append(r["v"])
    return {s: {i: float(np.mean(v)) for i, v in d.items()} for s, d in acc.items()}


def load_runs(roots, conds, step=None):
    runs = defaultdict(dict)
    for root in roots:
        for d in sorted(Path(root).glob("*/s*")):
            cond, seed = d.parent.name, int(d.name[1:])
            if conds and cond not in conds: continue
            if not ((d / "log.jsonl").exists() or (d / "log.jsonl.gz").exists()): continue
            log = read_log(d)
            if step is None:
                if not (d / "final.json").exists(): continue
                res, pp = json.loads((d / "final.json").read_text()).get("eval", {}), per_problem(d / "final_completions.jsonl.gz")
            else:
                ev = [r for r in log if r["kind"] == "eval" and r["step"] == step]
                if not ev or not (d / "evals" / f"step{step}.jsonl.gz").exists(): continue
                res, pp = {k: v for k, v in ev[0].items() if isinstance(v, dict)}, per_problem(d / "evals" / f"step{step}.jsonl.gz")
            if seed in runs[cond]: raise ValueError(f"{cond} s{seed} found in two roots")
            runs[cond][seed] = {"res": res, "log": log, "pp": pp}
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="", help="conditions to include, in this order (default: all found)")
    ap.add_argument("--baseline", default="rlvr")
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    conds = [c for c in a.conds.split(",") if c]
    runs = load_runs(a.root, conds, a.step)
    order = conds or [c for c in CONDS if c in runs] + sorted(c for c in runs if c not in CONDS)
    rep = {"compared_at": "final" if a.step is None else f"step {a.step}", "conditions": {}, "paired_vs_" + a.baseline: {}}
    for c in order:
        seeds = runs.get(c)
        if not seeds: continue
        e = {"seeds": sorted(seeds)}
        for s in SETS:
            xs = [r["res"][s]["acc"] for r in seeds.values() if s in r["res"]]
            if xs: e[s] = {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "per_seed": xs}
        m5 = [r["res"]["math500"] for r in seeds.values() if "math500" in r["res"]]
        if m5 and "kev" in m5[0]:
            e["math500_kev_correct_mean"] = float(np.mean([x["kev"]["correct"] for x in m5]))
            e["math500_kev_correct_auroc"] = float(np.mean([x["kev_correct_auroc"] for x in m5 if x["kev_correct_auroc"] is not None]))
        steps = [[r for r in s["log"] if r["kind"] == "step" and (a.step is None or r["step"] < a.step)] for s in seeds.values()]
        last = [st[-20:] for st in steps if st]
        if last:
            keys = ("acc", "reward_mean", "len_mean", "truncated", "groups_all_wrong", "groups_all_right", "lp_absdiff_mean", "tis_clip_frac", "gen_tokens")
            e["train_last20"] = {k: float(np.mean([np.mean([r[k] for r in st]) for st in last])) for k in keys if all(k in st[0] for st in last)}
            e["train_last20"]["kev_correct"] = float(np.mean([np.mean([r["kev"]["correct"] for r in st]) for st in last]))
            e["seconds_per_step"] = float(np.mean([np.mean([sum(r["seconds"].values()) for r in st]) for st in steps if st]))
            ex = {}   # per-step judge statistics of the label (III-B) and tree (III-C) conditions: seed mean over steps
            for k in EXTRA:
                xs = [[r[k] for r in st if r.get(k) is not None] for st in steps]
                xs = [x for x in xs if x]
                if xs: ex[k] = {w: float(np.mean([np.mean(sl(x)) for x in xs])) for w, sl in
                                (("all", lambda x: x), ("first20", lambda x: x[:20]), ("last20", lambda x: x[-20:]))}
            if ex: e["train_judge"] = ex
        rep["conditions"][c] = e
    base = runs.get(a.baseline, {})
    rng = np.random.default_rng(0)
    for c in order:
        seeds = runs.get(c)
        if c == a.baseline or not base or not seeds: continue
        rep["paired_vs_" + a.baseline][c] = {}
        for s in SETS:
            sets = [r["pp"].get(s, {}) for r in list(seeds.values()) + list(base.values())]
            ids = sorted(set.intersection(*[set(x) for x in sets])) if all(sets) else []
            if not ids: continue
            x = np.array([np.mean([r["pp"][s][i] for r in seeds.values()]) for i in ids])
            y = np.array([np.mean([r["pp"][s][i] for r in base.values()]) for i in ids])
            d = x - y
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            rep["paired_vs_" + a.baseline][c][s] = {"problems": len(ids), "diff": float(d.mean()),
                                                    "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    (out / "summary.json").write_text(json.dumps(rep, indent=1))

    os.environ["MPLBACKEND"] = "Agg"   # a notebook's inline backend (set by Colab) would fail outside the notebook
    import matplotlib.pyplot as plt
    keys = [("acc", "train accuracy (verifier)"), ("len_mean", "completion tokens"), ("groups_all_wrong", "groups with no correct sample"),
            ("kev_correct", "Kev P(correct)")]
    fig, axes = plt.subplots(1, len(keys) + 2, figsize=(4 * (len(keys) + 2), 3.2))
    for c in order:
        seeds = runs.get(c)
        if not seeds: continue
        for ax, (k, title) in zip(axes, keys):
            ys = [[r["kev"]["correct"] if k == "kev_correct" else r[k] for r in s["log"] if r["kind"] == "step"] for s in seeds.values()]
            n = min(map(len, ys)); w = 10
            if n < w: continue
            sm = np.array([np.convolve(y[:n], np.ones(w) / w, mode="valid") for y in ys])
            ax.plot(np.arange(sm.shape[1]) + w - 1, sm.mean(0), label=c); ax.set_title(title); ax.set_xlabel("GRPO step")
        for ax, sname in zip(axes[len(keys):], ("math500", "val")):
            ev = [[(r["step"], r[sname]["acc"]) for r in s["log"] if r["kind"] == "eval" and sname in r] for s in seeds.values()]
            n = min(map(len, ev))
            if n: ax.plot([x for x, _ in ev[0][:n]], np.mean([[y for _, y in e[:n]] for e in ev], 0), marker="o", label=c)
            ax.set_title(f"{sname} accuracy (greedy)"); ax.set_xlabel("GRPO step")
    axes[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
