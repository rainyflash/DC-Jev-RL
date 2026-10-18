"""The GRPO starting point, Qwen3.5-4B-Base without an adapter, on every evaluation set of dcjev.llm.grpo (periodic and
final), evaluated --repeats times through the rollout server: the base row of the result tables, and the spread of one
evaluation (greedy decoding is deterministic only up to the batch composition, and the sampled sets use the same seeds
as every run's final evaluation).

    python -m dcjev.llm.eval_base --out /content/work/runs/part2/base_eval      # rollout server on --port
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from dcjev.llm.data import load, train_val
from dcjev.llm.grpo import Rollouts, evaluate
from dcjev.llm.verify import Verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/content/work/data")
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--eval_max_tokens", type=int, default=2048)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _, val = train_val(a.data)
    gsm = load(f"{a.data}/gsm8k_test.jsonl")
    gsm_sub = [gsm[i] for i in sorted(np.random.default_rng(0).choice(len(gsm), 500, replace=False))]   # as grpo's gsm8k500
    sets = {"val": (val, 1, 0.0, 1.0), "math500": (load(f"{a.data}/math500.jsonl"), 1, 0.0, 1.0), "gsm8k500": (gsm_sub, 1, 0.0, 1.0),
            "gsm8k_test": (gsm, 1, 0.0, 1.0)}
    for name in ("amc23", "aime24", "aime25"):
        if Path(f"{a.data}/{name}.jsonl").exists(): sets[name] = (load(f"{a.data}/{name}.jsonl"), 16, 0.6, 0.95)
    from dcjev.llm.shared import SharedBackbone
    sb = SharedBackbone()   # Kev's view of the base policy on MATH-500, as in each run's final evaluation
    ro, ver = Rollouts(a.port), Verifier()
    for k in range(a.repeats):
        with gzip.open(out / f"eval{k}.jsonl.gz", "wt") as dump:
            res = evaluate(ro, None, sets, ver, a.eval_max_tokens, sb=sb, judge_sets=("math500",), dump=dump)
        (out / f"eval{k}.json").write_text(json.dumps(res, indent=1))
        print(k, {s: round(e["acc"], 4) for s, e in res.items()}, flush=True)


if __name__ == "__main__":
    main()
