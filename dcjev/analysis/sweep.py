"""Part II learning-rate sweep: validation accuracy per learning rate and the choice made by the rule of
experiments/part2_llm/LR_SELECTION.md (highest mean validation accuracy over steps 30 and 60; the smaller learning rate
when the best two are within 0.5 points).

    python -m dcjev.analysis.sweep --root runs_local/part2_sweep --out experiments/part2_llm/results/lr_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dcjev.analysis import read_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for d in sorted(Path(a.root).glob("lr*")):
        if not (d / "final.json").exists(): continue
        log = read_log(d)
        ev = {r["step"]: r["val"]["acc"] for r in log if r.get("kind") == "eval" and "val" in r}
        st = [r for r in log if r.get("kind") == "step"]
        runs[d.name[2:]] = {"val_acc": ev, "score": float(np.mean([ev[30], ev[60]])),
                            "train_acc_last10": float(np.mean([r["acc"] for r in st[-10:]])),
                            "grad_norm_mean": float(np.mean([r["grad_norm"] for r in st])),
                            "len_mean_last10": float(np.mean([r["len_mean"] for r in st[-10:]]))}
    order = sorted(runs, key=lambda k: -runs[k]["score"])
    best = order[0]
    if len(order) > 1 and runs[order[0]]["score"] - runs[order[1]]["score"] <= 0.005:
        best = min(order[:2], key=float)
    rep = {"rule": "argmax over learning rates of mean(val acc at steps 30 and 60); the smaller of the best two if within 0.5 points",
           "runs": runs, "chosen_lr": best}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
