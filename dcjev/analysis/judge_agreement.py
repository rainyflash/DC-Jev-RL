"""Part I offline judge agreement: re-score windows logged during the `env` runs with every judge and compare the scores
with the environment's own reward on the same windows.

    python -m dcjev.analysis.judge_agreement --root /content/work/runs/part1 --out experiments/part1_control/results/judge_agreement.json

The windows are the 2% sample every run logs (dcjev.control.ppo, type "window"), from all seeds and all of training, so
they cover failing, mediocre and good behaviour of the reference learner. Judges are queried through the running judge
server (dcjev.judges.server), with the prompts of the training runs. Reported per task: Spearman correlation of each
judge signal with the per-step environment reward of the window and with its mean forward velocity, and for the tasks
with falls the AUROC of p(healthy) for windows that did not end in a fall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dcjev.analysis.part1 import ENVS
from dcjev.judges.client import JudgeClient
from dcjev.metrics import auroc, ranks


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=6100)
    ap.add_argument("--max_per_env", type=int, default=4000)
    a = ap.parse_args()
    jc = JudgeClient(port=a.port)
    rng = np.random.default_rng(0)
    rep = {}
    for e in ENVS:
        wins = []
        for f in sorted(Path(a.root).glob(f"{e}/env/s*/log.jsonl")):
            wins += [json.loads(l) for l in open(f) if '"type": "window"' in l]
        if not wins: continue
        if len(wins) > a.max_per_env: wins = [wins[i] for i in sorted(rng.choice(len(wins), a.max_per_env, replace=False))]
        S = [w["summary"] for w in wins]
        per_step = np.array([w["env_reward_sum"] / w["len"] for w in wins])
        vx = np.array([w["summary"]["forward_velocity_m_per_s"]["mean"] for w in wins])
        k = jc.score("kev", e, S, ("progress", "healthy"))
        r = jc.score("ar", e, S, ("progress", "healthy"))
        sig = {"kev_progress": np.array(k["progress"]), "kev_healthy": np.array(k["healthy"]),
               "kev_gated": np.array(k["progress"]) * np.array(k["healthy"]),
               "ar_progress": np.array(r["progress"]), "ar_progress_sampled": np.array(r["progress_sampled"]),
               "ar_healthy": np.array(r["healthy"]), "ar_gated": np.array(r["progress"]) * np.array(r["healthy"])}
        out = {"windows": len(wins), "env_reward_per_step": {"mean": float(per_step.mean()), "std": float(per_step.std())},
               "ar_valid_mass": {q: float(np.mean(v)) for q, v in r.get("valid_mass", {}).items()}}
        for name, s in sig.items():
            out[name] = {"mean": float(s.mean()), "std": float(s.std()), "spearman_env_reward": spearman(s, per_step), "spearman_vx": spearman(s, vx)}
        fell = [w["summary"].get("fell_over") for w in wins]
        if any(x is not None for x in fell):
            ok = np.array([x == "no" for x in fell])
            out["healthy_auroc_no_fall"] = {"kev": auroc(sig["kev_healthy"], ok), "ar": auroc(sig["ar_healthy"], ok), "fall_rate": float(1 - ok.mean())}
        rep[e] = out
        print(e, json.dumps({n: out[n] for n in sig}, indent=0), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
