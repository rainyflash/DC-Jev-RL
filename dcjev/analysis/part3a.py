"""Part III-A aggregate: multi-turn code repair runs (dcjev.code.grpo_code).

    python -m dcjev.analysis.part3a --root /content/work/runs/part3a --out experiments/part3a_code/results

Per condition and evaluation set (final greedy evaluation; seed mean and spread): visible-test pass, all-tests pass
(visible and held-out), hack rate (visible passed, held-out failed), single-turn all-tests pass, attempts used; paired
per-problem differences of all-tests pass and hack against `vis` (bootstrap over problems); training curves; the spread
of the step-0 evaluations (the same base model in every run), which measures the evaluation noise.
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

CONDS = ["vis", "vis_kev", "hidden"]
SETS = ["kodcode_val", "mbppplus", "humanevalplus"]
METRICS = ["vis_pass", "all_pass", "hack", "hack_among_vis_pass", "first_attempt_all_pass", "attempts_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="vis")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = defaultdict(dict)
    for f in sorted(Path(a.root).glob("*/s*/final.json")):
        d = f.parent
        pp = defaultdict(dict)
        with gzip.open(d / "final_episodes.jsonl.gz", "rt") as g:
            for l in g:
                r = json.loads(l); pp[r["set"]][r["id"]] = (r["vis_pass"] and r["held_pass"], r["vis_pass"] and not r["held_pass"])
        runs[d.parent.name][int(d.name[1:])] = {"final": json.loads(f.read_text())["eval"], "log": read_log(d), "pp": pp}
    rep = {"conditions": {}, "paired_vs_" + a.baseline: {}}
    for c in [c for c in CONDS if c in runs]:
        e = {"seeds": sorted(runs[c])}
        for s in SETS:
            e[s] = {}
            for m in METRICS:
                xs = [r["final"][s][m] for r in runs[c].values() if s in r["final"] and r["final"][s].get(m) is not None]
                if xs: e[s][m] = {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "per_seed": xs}
            kv = [r["final"][s].get("kev_genuine_auroc_all_pass_given_vis") for r in runs[c].values() if s in r["final"]]
            kv = [x for x in kv if x is not None]
            if kv: e[s]["kev_genuine_auroc_all_pass_given_vis"] = float(np.mean(kv))
        steps = [[r for r in s["log"] if r["kind"] == "step"] for s in runs[c].values()]
        last = [st[-20:] for st in steps if st]
        e["train_last20"] = {k: float(np.mean([np.mean([r[k] for r in st]) for st in last])) for k in ("vis_pass", "all_pass", "hack", "attempts_mean", "reward_mean")}
        e["seconds_per_step"] = float(np.mean([np.mean([sum(r["seconds"].values()) for r in st]) for st in steps]))
        rep["conditions"][c] = e
    # Evaluation noise: every run's step-0 evaluation is the same base model (greedy decoding, different server states).
    s0 = defaultdict(list)
    for r in (r for c in runs for r in runs[c].values()):
        for ev in (x for x in r["log"] if x["kind"] == "eval" and x["step"] == 0):
            for s in SETS:
                for m in ("vis_pass", "all_pass", "hack"):
                    if isinstance(ev.get(s), dict) and ev[s].get(m) is not None: s0[(s, m)].append(ev[s][m])
    rep["step0_eval"] = {f"{s}/{m}": {"runs": len(v), "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                      "min": float(min(v)), "max": float(max(v))} for (s, m), v in s0.items()}
    base, rng = runs.get(a.baseline, {}), np.random.default_rng(0)
    for c in [c for c in CONDS if c in runs and c != a.baseline]:
        rep["paired_vs_" + a.baseline][c] = {}
        for s in SETS:
            ids = sorted(set.intersection(*[set(r["pp"][s]) for r in list(runs[c].values()) + list(base.values())]))
            if not ids: continue
            e = {"problems": len(ids)}
            for k, name in ((0, "all_pass"), (1, "hack")):
                d = np.array([np.mean([r["pp"][s][i][k] for r in runs[c].values()]) - np.mean([r["pp"][s][i][k] for r in base.values()]) for i in ids])
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
                e[name] = {"diff": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
            rep["paired_vs_" + a.baseline][c][s] = e
    (out / "summary.json").write_text(json.dumps(rep, indent=1))

    os.environ["MPLBACKEND"] = "Agg"   # a notebook's inline backend (set by Colab) would fail outside the notebook
    import matplotlib.pyplot as plt
    keys = [("vis_pass", "visible tests pass (train)"), ("all_pass", "all tests pass (train)"), ("hack", "visible pass, held-out fail (train)"),
            ("attempts_mean", "attempts per episode")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2))
    for c in [c for c in CONDS if c in runs]:
        for ax, (k, title) in zip(axes, keys):
            ys = [[r[k] for r in s["log"] if r["kind"] == "step"] for s in runs[c].values()]
            n = min(map(len, ys)); w = 10
            if n < w: continue
            sm = np.array([np.convolve(y[:n], np.ones(w) / w, mode="valid") for y in ys])
            ax.plot(np.arange(sm.shape[1]) + w - 1, sm.mean(0), label=c); ax.set_title(title); ax.set_xlabel("GRPO step")
    axes[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
