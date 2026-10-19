"""Check dcjev.judges.kev_judge.KevJudge against kev's reference path and measure its throughput.

    python -m dcjev.judges.check_kev_parity --out experiments/part1_control/kev_parity.json

For every request, KevJudge.probs (batched row form) is compared with DecisionModel.probs (kev's serving miss path, one
request at a time). Reported: max and mean absolute probability difference and argmax agreement, per dtype.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from dcjev.judges.kev_judge import KevJudge

LEVELS = ["Failing: fallen over, falling, or moving backward", "Poor: upright but hardly moving forward",
          "Fair: moving forward slowly", "Good: moving forward at a steady pace", "Excellent: moving forward fast and stably"]


def make_requests(n, seed=0):
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        kind = i % 3
        if kind == 0:   # MuJoCo-style telemetry state
            v = rng.uniform(-1, 4)
            state = {"task": "A one-legged robot should hop forward as fast as possible without falling.",
                     "forward_velocity_m_per_s": {"mean": round(v, 2), "min": round(v - rng.uniform(0, 1), 2), "max": round(v + rng.uniform(0, 1), 2)},
                     "torso_height_m": {"mean": round(rng.uniform(0.6, 1.4), 2), "min": round(rng.uniform(0.5, 1.3), 2)},
                     "torso_tilt_rad": {"max_abs": round(rng.uniform(0, 0.6), 2)},
                     "action_effort": round(rng.uniform(0, 1), 3),
                     "episode_ended": rng.choice(["yes", "no"])}
            qs = {"progress": {"type": "score", "instructions": "How well is the robot achieving the task during this segment?", "criteria": LEVELS},
                  "healthy": {"type": "noul", "instructions": "Is the robot moving forward with a genuine, stable gait?"}}
        elif kind == 1:   # free text, choice question
            state = rng.choice(["I was charged twice for order 1182. Please refund one of the charges.",
                                "The package arrived broken and I want a replacement as soon as possible.",
                                "Can you tell me when my order will ship? It has been a week.",
                                "My password reset link does not work and I cannot log in."]) + f" (ticket {i})"
            qs = {"team": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Charges and refunds", "shipping": "Deliveries", "returns": "Exchanges", "accounts": "Logins and passwords"}},
                  "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}}
        else:   # math-solution style state, score + noul
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            ans = a * b + rng.choice([0, 0, 1, -1, 10])
            state = {"problem": f"What is {a} times {b}?", "proposed_solution": f"{a} * {b} = {ans}. The answer is {ans}."}
            qs = {"correct": {"type": "noul", "instructions": "Is the proposed solution correct?"},
                  "quality": {"type": "score", "instructions": "How rigorous is the reasoning?", "criteria": ["wrong", "unclear", "plausible", "rigorous"]}}
        reqs.append({"state": state, "questions": qs})
    return reqs


def parity(judge, reqs):
    batched, _ = judge.probs(reqs)
    diffs, agree, total = [], 0, 0
    for r, b in zip(reqs, batched):
        ref = judge.reference_probs(r)
        for pq, rq in zip(b, ref):
            d = np.abs(np.array(pq) - np.array(rq))
            diffs.append(float(d.max()))
            agree += int(np.argmax(pq) == np.argmax(rq)); total += 1
    return {"questions": total, "max_abs_diff": max(diffs), "mean_max_abs_diff": float(np.mean(diffs)),
            "p99_max_abs_diff": float(np.quantile(diffs, 0.99)), "argmax_agreement": agree / total}


def throughput(judge, reqs, budgets):
    res = []
    for tb in budgets:
        judge.token_budget = tb
        judge.probs(reqs[:64])   # warm-up (kernel autotuning)
        torch.cuda.synchronize(); t = time.perf_counter()
        judge.probs(reqs)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        res.append({"token_budget": tb, "requests": len(reqs), "seconds": dt, "requests_per_s": len(reqs) / dt})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--throughput_n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reqs = make_requests(a.n)
    report = {"run": a.run, "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    for dtype in ["fp32", "bf16"]:
        judge = KevJudge(a.run, dtype=dtype)
        report[f"parity_{dtype}"] = parity(judge, reqs)
        report["temperature"] = judge.temperature
        if dtype == "bf16":
            tp_reqs = [r for r in make_requests(a.throughput_n, seed=1) if "task" in str(r["state"])]
            report["throughput_bf16_telemetry"] = throughput(judge, tp_reqs, [16384, 32768, 65536])
            tok = judge.tok
            enc, _ = judge.encode(tp_reqs[0])
            report["example_tokens_per_request"] = len(enc["ids"])
        print(json.dumps(report, indent=1), flush=True)
        del judge; torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
