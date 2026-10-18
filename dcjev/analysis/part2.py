"""Part II aggregate: final accuracies per condition (seed mean and spread), paired per-problem differences against
RLVR with bootstrap intervals, training curves, and the judge's view of each trained policy.

    python -m dcjev.analysis.part2 --root /content/work/runs/part2 --out experiments/part2_llm/results

Paired differences: for each benchmark, per-problem accuracy is averaged over the seeds of a condition (greedy sets: the
fraction of seeds that solve it; sampled sets: the mean over seeds x samples); the difference to RLVR is averaged over
problems, with a bootstrap over problems (10k resamples) for the 95% interval. Seed-level spread is reported alongside.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ["rlvr", "rlvr_kev_add", "rlvr_kev_gate", "kev_only", "rlvr_ar_add"]
SETS = ["gsm8k_test", "math500", "amc23", "aime24", "aime25"]


def per_problem(run_dir):
    acc = defaultdict(lambda: defaultdict(list))
    with gzip.open(run_dir / "final_completions.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l); acc[r["set"]][r["id"]].append(r["v"])
    return {s: {i: float(np.mean(v)) for i, v in d.items()} for s, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="rlvr")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = defaultdict(dict)
    for f in sorted(root.glob("*/s*/final.json")):
        cond, seed = f.parent.parent.name, int(f.parent.name[1:])
        log = [json.loads(l) for l in open(f.parent / "log.jsonl")]
        runs[cond][seed] = {"final": json.loads(f.read_text()).get("eval", {}), "log": log, "pp": per_problem(f.parent)}
    rep = {"conditions": {}, "paired_vs_" + a.baseline: {}}
    for c, seeds in runs.items():
        e = {"seeds": sorted(seeds)}
        for s in SETS:
            xs = [r["final"][s]["acc"] for r in seeds.values() if s in r["final"]]
            if xs: e[s] = {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "per_seed": xs}
        m5 = [r["final"]["math500"] for r in seeds.values() if "math500" in r["final"]]
        if m5 and "kev" in m5[0]:
            e["math500_kev_correct_mean"] = float(np.mean([x["kev"]["correct"] for x in m5]))
            e["math500_kev_correct_auroc"] = float(np.mean([x["kev_correct_auroc"] for x in m5 if x["kev_correct_auroc"] is not None]))
        steps = [[r for r in s["log"] if r["kind"] == "step"] for s in seeds.values()]
        last = [st[-20:] for st in steps if st]
        if last:
            e["train_last20"] = {k: float(np.mean([np.mean([r[k] for r in st]) for st in last]))
                                 for k in ("acc", "reward_mean", "len_mean", "truncated", "groups_all_wrong", "groups_all_right", "lp_absdiff_mean", "tis_clip_frac")}
            e["train_last20"]["kev_correct"] = float(np.mean([np.mean([r["kev"]["correct"] for r in st]) for st in last]))
            e["seconds_per_step"] = float(np.mean([np.mean([sum(r["seconds"].values()) for r in st]) for st in steps if st]))
        rep["conditions"][c] = e
    base = runs.get(a.baseline, {})
    rng = np.random.default_rng(0)
    for c, seeds in runs.items():
        if c == a.baseline or not base: continue
        rep["paired_vs_" + a.baseline][c] = {}
        for s in SETS:
            ids = sorted(set.intersection(*[set(r["pp"].get(s, {})) for r in list(seeds.values()) + list(base.values())]))
            if not ids: continue
            x = np.array([np.mean([r["pp"][s][i] for r in seeds.values()]) for i in ids])
            y = np.array([np.mean([r["pp"][s][i] for r in base.values()]) for i in ids])
            d = x - y
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            rep["paired_vs_" + a.baseline][c][s] = {"problems": len(ids), "diff": float(d.mean()),
                                                    "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    (out / "summary.json").write_text(json.dumps(rep, indent=1))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("acc", "train accuracy (verifier)"), ("len_mean", "completion tokens"), ("groups_all_wrong", "groups with no correct sample"),
            ("kev_correct", "Kev P(correct)")]
    fig, axes = plt.subplots(1, len(keys) + 2, figsize=(4 * (len(keys) + 2), 3.2))
    for c, seeds in runs.items():
        for ax, (k, title) in zip(axes, keys):
            ys = []
            for s in seeds.values():
                st = [r for r in s["log"] if r["kind"] == "step"]
                ys.append([r["kev"]["correct"] if k == "kev_correct" else r[k] for r in st])
            n = min(map(len, ys)); Y = np.array([y[:n] for y in ys])
            w = 10; sm = np.array([np.convolve(y, np.ones(w) / w, mode="valid") for y in Y])
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
