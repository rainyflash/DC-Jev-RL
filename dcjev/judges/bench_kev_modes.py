"""Accuracy and throughput of KevJudge's execution modes against kev's fp32 reference path.

    python -m dcjev.judges.bench_kev_modes --out experiments/part1_control/kev_modes.json

Reference: DecisionModel.probs in fp32 (kev's exact path), one request at a time. Modes (bf16): row form or shared
prefix, with or without kev.fused_qwen35. Reported per mode: max / p99 / mean absolute probability difference from the
fp32 reference, argmax agreement, and requests per second on MuJoCo-telemetry requests (2 questions each).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from dcjev.judges.check_kev_parity import make_requests
from dcjev.judges.kev_judge import KevJudge


def compare(probs, ref):
    d, agree, n = [], 0, 0
    for b, r in zip(probs, ref):
        for pq, rq in zip(b, r):
            d.append(float(np.abs(np.array(pq) - np.array(rq)).max())); agree += int(np.argmax(pq) == np.argmax(rq)); n += 1
    return {"max_abs_diff": max(d), "p99_abs_diff": float(np.quantile(d, 0.99)), "mean_abs_diff": float(np.mean(d)), "argmax_agreement": agree / n, "questions": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--tp_n", type=int, default=2000)
    a = ap.parse_args()
    reqs = make_requests(a.n)
    tp = [r for r in make_requests(3 * a.tp_n, seed=1) if "task" in str(r["state"])][: a.tp_n]
    ref_judge = KevJudge(a.run, dtype="fp32")
    ref = [ref_judge.reference_probs(r) for r in reqs]
    del ref_judge; torch.cuda.empty_cache()
    report = {"run": a.run, "reference": "kev DecisionModel.probs, fp32", "modes": []}
    for shared in (False, True):
        for fused in (False, True):
            name = f"{'shared_prefix' if shared else 'row_form'}{'+fused' if fused else ''}"
            try:
                j = KevJudge(a.run, dtype="bf16", shared_prefix=shared, fused=fused, token_budget=32768)
                acc = compare(j.probs(reqs)[0], ref)
                j.probs(tp[:128])
                torch.cuda.synchronize(); t = time.perf_counter(); j.probs(tp); torch.cuda.synchronize()
                dt = time.perf_counter() - t
                rec = {"mode": name, **acc, "requests_per_s": len(tp) / dt, "seconds": dt}
                del j; torch.cuda.empty_cache()
            except Exception as e:
                rec = {"mode": name, "error": repr(e)[:500]}
            print(json.dumps(rec), flush=True)
            report["modes"].append(rec)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
