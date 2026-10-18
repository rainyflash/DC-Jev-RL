"""Part I aggregate: final returns, normalised scores with bootstrap intervals, learning curves, behaviour, judge cost.

    python -m dcjev.analysis.part1 --root /content/work/runs/part1 --out experiments/part1_control/results

Every number comes from the runs' final.json / log.jsonl. Final return = mean undiscounted environment return of the
deterministic policy over 20 evaluation episodes after 1M steps. Normalised score per task = (R - R_random) /
(R_env - R_random), with R_env the mean final return of the `env` condition over its seeds and R_random the mean return
of a uniform random policy (20 episodes, measured here). Aggregates over the 5 tasks x seeds use the interquartile
mean (IQM) with stratified bootstrap 95% intervals (Agarwal et al., 2021).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dcjev.analysis import read_log

CONDS = ["env", "env_window", "kev_score", "kev_gated", "kev_distill", "ar_gated", "ar_sampled"]
ENVS = ["Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Swimmer-v5", "Reacher-v5"]


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    if n == 0: return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def stratified_bootstrap(groups, stat, reps=10000, seed=0):
    """groups: list of arrays (one per task); resample seeds within each task, apply stat to the pooled sample."""
    rng = np.random.default_rng(seed)
    vals = [stat(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(reps)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def prob_improvement(a_groups, b_groups):
    """P(X > Y) for a random run X of condition a and Y of b on the same task, averaged over tasks (ties count 1/2)."""
    ps = []
    for a, b in zip(a_groups, b_groups):
        a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
        ps.append(float(((a > b) + 0.5 * (a == b)).mean()))
    return float(np.mean(ps))


def random_returns(env_id, episodes=20, seed=30000):
    from dcjev.control.envs import make_env
    env, rets = make_env(env_id, seed), []
    rng = np.random.default_rng(seed)
    for e in range(episodes):
        env.reset(seed=seed + e); done, R = False, 0.0
        while not done:
            _, r, term, trunc, _ = env.step(rng.uniform(env.action_space.low, env.action_space.high))
            R += r; done = term or trunc
        rets.append(R)
    return float(np.mean(rets))


def load_runs(root):
    runs = []
    for f in sorted(Path(root).glob("*/*/s*/final.json")):
        j = json.loads(f.read_text())
        log = read_log(f.parent)
        j["curve"] = [(r["step"], r["return_mean"]) for r in log if r["type"] == "eval"]
        j["corr"] = [r.get("corr_window_score_env") for r in log if r["type"] == "update"]
        runs.append(j)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.root)
    conds = [c for c in CONDS if any(r["reward"] == c for r in runs)]
    by = {(e, c): sorted([r for r in runs if r["env_id"] == e and r["reward"] == c], key=lambda r: r["seed"]) for e in ENVS for c in conds}
    rand = {e: random_returns(e) for e in ENVS}
    ref = {e: float(np.mean([r["final_return_mean"] for r in by[(e, "env")]])) for e in ENVS if by[(e, "env")]}

    rows = []
    for r in runs:
        rows.append({k: r.get(k) for k in ("env_id", "reward", "seed", "final_return_mean", "final_return_std", "final_len_mean",
                                           "final_vx_mean", "final_still_frac", "final_final_dist_mean", "wall_s", "judge_windows", "judge_wait_s")})
    with open(out / "final_returns.csv", "w") as f:
        keys = list(rows[0])
        f.write(",".join(keys) + "\n")
        for r in rows: f.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")

    summary = {"random_policy_return": rand, "env_reference_return": ref, "per_task": {}, "aggregate": {}}
    for e in ENVS:
        summary["per_task"][e] = {}
        for c in conds:
            x = np.array([r["final_return_mean"] for r in by[(e, c)]])
            if len(x) == 0: continue
            summary["per_task"][e][c] = {
                "seeds": [r["seed"] for r in by[(e, c)]], "returns": x.tolist(), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else None,
                "median": float(np.median(x)), "ci95_mean": stratified_bootstrap([x], np.mean) if len(x) > 1 else None,
                "still_frac": float(np.mean([r["final_still_frac"] for r in by[(e, c)]])),
                "vx_mean": float(np.mean([r["final_vx_mean"] for r in by[(e, c)]])),
                "wall_s_mean": float(np.mean([r["wall_s"] for r in by[(e, c)]])),
                "judge_windows_mean": float(np.mean([r["judge_windows"] for r in by[(e, c)]])),
                "judge_wait_s_mean": float(np.mean([r["judge_wait_s"] for r in by[(e, c)]])),
                "mean_corr_window_score_env": float(np.nanmean([v for r in by[(e, c)] for v in r["corr"] if v is not None])) if c not in ("env", "env_window") else None}
    norm = lambda e, x: (np.asarray(x) - rand[e]) / (ref[e] - rand[e])
    groups = {c: [norm(e, [r["final_return_mean"] for r in by[(e, c)]]) for e in ENVS if by[(e, c)] and e in ref] for c in conds}
    for c in conds:
        g = groups[c]
        if len(g) < len(ENVS): continue
        pooled = np.concatenate(g)
        summary["aggregate"][c] = {"runs": len(pooled), "iqm_normalised": iqm(pooled), "iqm_ci95": stratified_bootstrap(g, iqm),
                                   "mean_normalised": float(pooled.mean()), "mean_ci95": stratified_bootstrap(g, np.mean),
                                   "median_normalised": float(np.median(pooled)),
                                   "p_improvement_vs_env": prob_improvement(g, groups["env"]),
                                   "p_improvement_vs_env_window": prob_improvement(g, groups["env_window"]) if "env_window" in groups else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(ENVS), figsize=(4 * len(ENVS), 3.2))
    for ax, e in zip(axes, ENVS):
        for c in conds:
            cs = [r["curve"] for r in by[(e, c)] if r["curve"]]
            if not cs: continue
            n = min(len(x) for x in cs)
            steps = np.array([s for s, _ in cs[0][:n]]); Y = np.array([[v for _, v in x[:n]] for x in cs])
            m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else np.zeros(n)
            ax.plot(steps / 1e6, m, label=c); ax.fill_between(steps / 1e6, m - se, m + se, alpha=0.15)
        ax.set_title(e); ax.set_xlabel("environment steps (M)")
    axes[0].set_ylabel("evaluation return (mean ± s.e. over seeds)"); axes[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=150)
    print(json.dumps(summary["aggregate"], indent=1))


if __name__ == "__main__":
    main()
