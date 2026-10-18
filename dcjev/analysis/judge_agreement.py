"""Part I offline judge agreement: re-score windows logged during the `env` runs with every judge and compare the scores
with the environment's own reward on the same windows.

    python -m dcjev.analysis.judge_agreement --root /content/work/runs/part1 --out experiments/part1_control/results/judge_agreement.json
    python -m dcjev.analysis.judge_agreement --root ... --out ... --from_scores    # statistics only, from the saved scores

The windows are the 2% sample every run logs (dcjev.control.ppo, type "window"), from all seeds and all of training, so
they cover failing, mediocre and good behaviour of the reference learner. Judges are queried through the running judge
server (dcjev.judges.server), with the prompts of the training runs. Reported per task: Spearman correlation of each
judge signal with the per-step environment reward of the window and with the physical quantity the task is about (the
mean forward velocity; for Reacher the reduction of the fingertip-target distance over the window), and for the tasks
with falls the AUROC of p(healthy) for windows that did not end in a fall. 95% intervals for the correlations and for
the differences between judges come from a two-stage bootstrap: resample the runs (seeds), then the windows of each
resampled run, so that windows of one run are not treated as independent. Every window's scores are saved next to the
output (<out>_scores.jsonl.gz). Each task's result is written as soon as it is done, and a task already in the output
file is not scored again.
"""
from __future__ import annotations

import argparse
import gzip
import json
import zlib
from pathlib import Path

import numpy as np

from dcjev.analysis import read_log
from dcjev.analysis.part1 import ENVS
from dcjev.metrics import auroc, ranks

SIGNALS = ("kev_progress", "kev_healthy", "kev_gated", "ar_progress", "ar_progress_sampled", "ar_healthy", "ar_gated")
PAIRS = (("kev_progress", "ar_progress"), ("kev_gated", "ar_gated"), ("kev_healthy", "ar_healthy"), ("ar_progress", "ar_progress_sampled"))


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def score(jc, judge, env, summaries, questions, chunk=256):
    """Scores in chunks, so that the training runs sharing the judge server are not held up behind one large batch."""
    out = {}
    for i in range(0, len(summaries), chunk):
        r = jc.score(judge, env, summaries[i:i + chunk], questions)
        for k, v in r.items():
            if isinstance(v, dict):
                for q, x in v.items(): out.setdefault(k, {}).setdefault(q, []).extend(x)
            else:
                out.setdefault(k, []).extend(v)
    return out


def cluster_bootstrap(rows, n=2000, seed=0):
    """95% intervals of each signal's Spearman correlation (with the per-step env reward and with the physical quantity)
    and of the differences of PAIRS, resampling runs and then the windows within each resampled run."""
    rng = np.random.default_rng(seed)
    by_run = {}
    for i, r in enumerate(rows): by_run.setdefault(r["seed"], []).append(i)
    runs = [np.array(v) for v in by_run.values()]
    cols = {k: np.array([r[k] for r in rows], dtype=float) for k in SIGNALS + ("per_step", "phys")}
    boots = {t: {k: [] for k in SIGNALS + tuple(f"{a}-{b}" for a, b in PAIRS)} for t in ("per_step", "phys")}
    for _ in range(n):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in (runs[j] for j in rng.integers(0, len(runs), len(runs)))])
        for t in ("per_step", "phys"):
            rho = {k: spearman(cols[k][idx], cols[t][idx]) for k in SIGNALS}
            for k in SIGNALS: boots[t][k].append(np.nan if rho[k] is None else rho[k])
            for a, b in PAIRS: boots[t][f"{a}-{b}"].append(np.nan if None in (rho[a], rho[b]) else rho[a] - rho[b])
    q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
    out = {"runs": len(runs), "resamples": n}
    for t, key in (("per_step", "env_reward"), ("phys", "physical")):
        out[f"ci95_{key}"] = {k: q(v) for k, v in boots[t].items()}
        out[f"diff_{key}"] = {f"{a}-{b}": (None if None in (spearman(cols[a], cols[t]), spearman(cols[b], cols[t]))
                                           else spearman(cols[a], cols[t]) - spearman(cols[b], cols[t])) for a, b in PAIRS}
    return out


def stats(rows):
    per_step, phys = np.array([r["per_step"] for r in rows]), np.array([r["phys"] for r in rows])
    out = {"windows": len(rows), "env_reward_per_step": {"mean": float(per_step.mean()), "std": float(per_step.std())}}
    vm = [r["ar_valid_mass"] for r in rows if r.get("ar_valid_mass")]
    if vm: out["ar_valid_mass"] = {q: float(np.mean([v[q] for v in vm])) for q in vm[0]}
    for name in SIGNALS:
        s = np.array([r[name] for r in rows])
        out[name] = {"mean": float(s.mean()), "std": float(s.std()), "spearman_env_reward": spearman(s, per_step), "spearman_physical": spearman(s, phys)}
    fell = [r["fell"] for r in rows]
    if any(x is not None for x in fell):
        ok = np.array([x == "no" for x in fell])
        out["healthy_auroc_no_fall"] = {"kev": auroc([r["kev_healthy"] for r in rows], ok), "ar": auroc([r["ar_healthy"] for r in rows], ok),
                                        "fall_rate": float(1 - ok.mean())}
    out["bootstrap"] = cluster_bootstrap(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=6100)
    ap.add_argument("--max_per_env", type=int, default=4000)
    ap.add_argument("--from_scores", action="store_true", help="recompute the statistics from the saved window scores")
    a = ap.parse_args()
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    sc_path = out_path.with_name(out_path.stem + "_scores.jsonl.gz")
    saved = {}
    if sc_path.exists():
        with gzip.open(sc_path, "rt") as f:
            for l in f:
                r = json.loads(l); saved.setdefault(r["task"], []).append(r)
    rep = {} if a.from_scores or not out_path.exists() else json.loads(out_path.read_text())
    jc = None
    for e in ENVS:
        if e in rep: continue
        if e not in saved:
            if a.from_scores: continue
            from dcjev.judges.client import JudgeClient
            jc = jc or JudgeClient(port=a.port)
            rng = np.random.default_rng(zlib.crc32(e.encode()))
            wins = []
            for d in sorted(Path(a.root).glob(f"{e}/env/s*")):
                wins += [{**r, "seed": int(d.name[1:])} for r in read_log(d) if r["type"] == "window"]
            if not wins: continue
            if len(wins) > a.max_per_env: wins = [wins[i] for i in sorted(rng.choice(len(wins), a.max_per_env, replace=False))]
            S = [w["summary"] for w in wins]
            k = score(jc, "kev", e, S, ("progress", "healthy"))
            r = score(jc, "ar", e, S, ("progress", "healthy"))
            rows = []
            for i, w in enumerate(wins):
                if e == "Reacher-v5":
                    phys = w["summary"]["fingertip_to_target_distance_m"]["start"] - w["summary"]["fingertip_to_target_distance_m"]["end"]
                else:
                    phys = w["summary"]["forward_velocity_m_per_s"]["mean"]
                kp, kh, ap_, ah = k["progress"][i], k["healthy"][i], r["progress"][i], r["healthy"][i]
                rows.append({"task": e, "seed": w["seed"], "step": w["step"], "per_step": w["env_reward_sum"] / w["len"], "phys": phys,
                             "fell": w["summary"].get("fell_over"), "kev_progress": kp, "kev_healthy": kh, "kev_gated": kp * kh,
                             "ar_progress": ap_, "ar_progress_sampled": r["progress_sampled"][i], "ar_healthy": ah, "ar_gated": ap_ * ah,
                             "ar_valid_mass": {q: v[i] for q, v in r.get("valid_mass", {}).items()}})
            with gzip.open(sc_path, "at") as f:
                for row in rows: f.write(json.dumps(row) + "\n")
            saved[e] = rows
        out = stats(saved[e])
        out["physical_quantity"] = "fingertip-target distance reduction (m)" if e == "Reacher-v5" else "mean forward velocity (m/s)"
        rep[e] = out
        out_path.write_text(json.dumps(rep, indent=1))
        print(e, json.dumps({n: out[n] for n in SIGNALS}, indent=0), flush=True)


if __name__ == "__main__":
    main()
