"""Part I: what the final policies do, beyond their return.

    python -m dcjev.analysis.behaviour --root /content/work/runs/part1 --out experiments/part1_control/results/behaviour.json

Each run's final policy (policy.pt) is run again on the 20 final-evaluation episodes, as dcjev.control.ppo.evaluate
runs it: the same seeds (20,000 + k), the deterministic action and the frozen observation statistics. Every step's
telemetry is recorded. The re-run's mean return is compared with the run's final.json, and the largest difference is
reported. Per run, and per task and condition, the script reports four quantities, each averaged over episodes. The
first is the fraction of steps with |forward velocity| < 0.1 m/s (still, as in evaluate). The second, for HalfCheetah,
is the fraction of steps with the torso upside down (cos(pitch) < 0, flipped). The third is the mean control effort,
the mean squared action clipped to the action bounds, in [0, 1]. The fourth, for Reacher, is the mean
fingertip-target distance over the episode. The same quantities are reported for the uniform random policy of
dcjev.analysis.part1 (same seeds, so its mean return equals the reference used for normalisation).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dcjev.analysis.part1 import CONDS, ENVS


def metrics(env_id, eps):
    """eps: one [steps, 5] telemetry array per episode (dcjev.control.envs.TEL); each quantity is averaged per episode."""
    per_ep = lambda f: float(np.mean([f(T) for T in eps]))
    out = {"still_frac": per_ep(lambda T: np.mean(np.abs(T[:, 0]) < 0.1)), "effort_mean": per_ep(lambda T: T[:, 4].mean())}
    if env_id == "HalfCheetah-v5": out["flipped_frac"] = per_ep(lambda T: np.mean(np.cos(T[:, 2]) < 0))
    if env_id == "Reacher-v5": out["dist_mean"] = per_ep(lambda T: T[:, 3].mean())
    return out


def rerun(run_dir, episodes=20, seed0=20_000):
    import torch
    from dcjev.control.envs import make_env
    from dcjev.control.ppo import Agent
    torch.set_num_threads(1)
    fin = json.loads((Path(run_dir) / "final.json").read_text())
    env_id = fin["env_id"]
    ck = torch.load(Path(run_dir) / "policy.pt", weights_only=False)   # our own checkpoint (numpy statistics inside)
    env = make_env(env_id, seed0)
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0])
    agent.load_state_dict(ck["agent"]); agent.eval()
    mean, var = ck["obs_mean"], ck["obs_var"]
    rets, eps = [], []
    for k in range(episodes):
        obs, info = env.reset(seed=seed0 + k)
        done, ret, tel = False, 0.0, []
        while not done:
            x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10)
            with torch.no_grad():
                a = agent.actor_mean(torch.tensor(x, dtype=torch.float32)[None]).numpy()[0]
            obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
            ret += float(r); done = term or trunc; tel.append(info["tel"])
        rets.append(ret); eps.append(np.array(tel))
    env.close()
    return {"env_id": env_id, "reward": fin["reward"], "seed": fin["seed"], "return_mean": float(np.mean(rets)),
            "final_json_return_mean": fin["final_return_mean"], **metrics(env_id, eps)}


def random_policy(env_id, episodes=20, seed=30000):
    """The uniform random policy of dcjev.analysis.part1.random_returns, with the same seeds."""
    from dcjev.control.envs import make_env
    env, rng, rets, eps = make_env(env_id, seed), np.random.default_rng(seed), [], []
    for e in range(episodes):
        env.reset(seed=seed + e)
        done, R, tel = False, 0.0, []
        while not done:
            _, r, term, trunc, info = env.step(rng.uniform(env.action_space.low, env.action_space.high))
            R += r; done = term or trunc; tel.append(info["tel"])
        rets.append(R); eps.append(np.array(tel))
    env.close()
    return {"env_id": env_id, "return_mean": float(np.mean(rets)), **metrics(env_id, eps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    dirs = sorted(str(p.parent) for p in Path(a.root).glob("*/*/s*/policy.pt"))
    with ProcessPoolExecutor(a.workers) as ex:
        runs = list(ex.map(rerun, dirs))
        rand = dict(zip(ENVS, ex.map(random_policy, ENVS)))
    agg = {}
    for e in ENVS:
        for c in CONDS:
            rs = [r for r in runs if r["env_id"] == e and r["reward"] == c]
            if not rs: continue
            d = {"runs": len(rs)}
            for k in ("still_frac", "effort_mean", "flipped_frac", "dist_mean"):
                v = [r[k] for r in rs if k in r]
                if v: d[k] = float(np.mean(v))
            if e == "HalfCheetah-v5": d["runs_flipped"] = int(sum(r["flipped_frac"] > 0.5 for r in rs))
            agg.setdefault(e, {})[c] = d
        agg.setdefault(e, {})["random"] = rand[e]
    rep = {"runs": len(runs), "max_abs_return_diff": max(abs(r["return_mean"] - r["final_json_return_mean"]) for r in runs),
           "per_task": agg, "per_run": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("runs", "max_abs_return_diff", "per_task")}, indent=1))


if __name__ == "__main__":
    main()
