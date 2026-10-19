"""Agreement of the fused autoregressive judge with the unfused one, and its throughput.

    python -m dcjev.judges.check_ar --out experiments/part1_control/ar_fused.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from dcjev.control import prompts
from dcjev.control.envs import ENVS, WINDOW, window_summary
from dcjev.judges.ar_judge import ARJudge

VMAX = {"Hopper-v5": 3.5, "Walker2d-v5": 4.5, "HalfCheetah-v5": 10.0, "Swimmer-v5": 0.4, "Reacher-v5": 0.0}
DT = {"Hopper-v5": 0.008, "Walker2d-v5": 0.008, "HalfCheetah-v5": 0.05, "Swimmer-v5": 0.04, "Reacher-v5": 0.02}


def synthetic_windows(n, seed=0):
    """Random but physically plausible window telemetry for every environment."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        e = ENVS[i % len(ENVS)]; H = WINDOW[e]
        tel = np.full((H, 5), np.nan)
        v = rng.uniform(-0.3 * VMAX[e], VMAX[e]) if VMAX[e] else 0.0
        tel[:, 0] = v + rng.normal(0, 0.2 * max(VMAX[e], 0.1), H)
        if e in ("Hopper-v5", "Walker2d-v5"):
            tel[:, 1] = rng.uniform(0.7, 1.3) + rng.normal(0, 0.03, H); tel[:, 2] = rng.normal(0, rng.uniform(0.02, 0.5), H)
        if e == "HalfCheetah-v5":
            tel[:, 2] = rng.normal(rng.choice([0.0, 0.0, 2.5]), 0.2, H)
        if e == "Reacher-v5":
            d0 = rng.uniform(0.0, 0.25); tel[:, 3] = np.linspace(d0, d0 * rng.uniform(0.0, 1.2), H)
        tel[:, 4] = rng.uniform(0, 1, H) ** 2
        ended = bool(e in ("Hopper-v5", "Walker2d-v5") and rng.random() < 0.15)
        out.append((e, window_summary(e, tel, ended, DT[e])))
    return out


def run(judge, wins):
    res = {}
    for q, kind in (("progress", "score"), ("healthy", "noul")):
        res[q] = judge.score([prompts.ar_messages(e, s, q) for e, s in wins], kind)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tp_n", type=int, default=2000)
    a = ap.parse_args()
    wins = synthetic_windows(a.n)
    ref = run(ARJudge(revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", fused=False), wins)
    torch.cuda.empty_cache()
    j = ARJudge(revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", fused=True)
    fz = run(j, wins)
    d_prog = np.abs(np.array([r["expected"] for r in fz["progress"]]) - np.array([r["expected"] for r in ref["progress"]]))
    d_yes = np.abs(np.array([r["p_yes"] for r in fz["healthy"]]) - np.array([r["p_yes"] for r in ref["healthy"]]))
    arg = np.mean([np.argmax(x["probs"]) == np.argmax(y["probs"]) for x, y in zip(fz["progress"], ref["progress"])])
    tp = synthetic_windows(a.tp_n, seed=1)
    run(j, tp[:64]); torch.cuda.synchronize(); t = time.perf_counter(); run(j, tp); torch.cuda.synchronize()
    dt = time.perf_counter() - t
    rep = {"n": a.n, "progress_expected_abs_diff": {"max": float(d_prog.max()), "mean": float(d_prog.mean())},
           "healthy_p_yes_abs_diff": {"max": float(d_yes.max()), "mean": float(d_yes.mean())}, "progress_argmax_agreement": float(arg),
           "valid_mass_progress_mean": float(np.mean([r["valid_mass"] for r in ref["progress"]])),
           "valid_mass_healthy_mean": float(np.mean([r["valid_mass"] for r in ref["healthy"]])),
           "fused_windows_per_s_two_questions": a.tp_n / dt}
    print(json.dumps(rep, indent=1))
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
