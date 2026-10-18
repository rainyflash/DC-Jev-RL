"""PPO with pluggable reward sources (a single-environment port of CleanRL's ppo_continuous_action.py, MIT licence,
with its default hyperparameters).

    python -m dcjev.control.ppo --env_id Hopper-v5 --reward kev_gated --seed 1 --out runs/part1/Hopper-v5/kev_gated/s1

Reward sources (--reward):
  env          the environment's own per-step reward (the engineered reward: reference condition)
  env_window   the environment's reward averaged over each judge window and given to every step of that window
               (controls for the coarser, window-level feedback the judges give)
  kev_score    Kev-4B's progress score in [0, 1] for each window
  kev_gated    Kev-4B's progress score multiplied by Kev-4B's p(healthy) (multiplicative Noul gate)
  kev_distill  a small reward network trained online on kev_gated labels of a fraction of the windows
               (confidence-weighted), used for the other windows
  ar_gated     the autoregressive judge (Qwen3.5-4B, expected-value readout): progress x p(healthy)
  ar_sampled   the autoregressive judge with a sampled discrete progress level x p(healthy)
Every source goes through the same reward normalisation (gymnasium's NormalizeReward, reproduced step by step) and
clipping. The policy is always evaluated on the environment's own return, which no judge ever sees.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from dcjev.control.envs import WINDOW, make_env, window_features, window_summary

REWARDS = ["env", "env_window", "kev_score", "kev_gated", "kev_distill", "ar_gated", "ar_sampled"]


class RunningMeanStd:
    """gymnasium.wrappers.utils.RunningMeanStd (parallel-algorithm update)."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean, self.var, self.count = np.zeros(shape, np.float64), np.ones(shape, np.float64), epsilon

    def update(self, x):
        x = np.asarray(x, np.float64)
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        new_mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + np.square(delta) * self.count * bc / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(),
                                    layer_init(nn.Linear(64, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(),
                                        layer_init(nn.Linear(64, act_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


class Distiller:
    """Reward network for kev_distill: window features -> judge label, confidence-weighted MSE, trained online."""

    def __init__(self, n_in, seed, lr=1e-3):
        g = torch.Generator().manual_seed(seed + 12345)
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) * (1.0 / np.sqrt(m.in_features))); m.bias.zero_()
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.X, self.y, self.w = [], [], []
        self.rms = RunningMeanStd((n_in,))
        self.rng = np.random.default_rng(seed + 999)

    def add(self, X, y, w):
        self.X.append(X); self.y.append(y); self.w.append(w); self.rms.update(X)

    def _norm(self, X):
        return torch.tensor((X - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), dtype=torch.float32)

    def train(self, steps=200, batch=256):
        X, y, w = np.concatenate(self.X), np.concatenate(self.y), np.concatenate(self.w)
        Xt, yt, wt = self._norm(X), torch.tensor(y, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)
        for _ in range(steps):
            idx = torch.as_tensor(self.rng.integers(0, len(X), size=min(batch, len(X))))
            pred = torch.sigmoid(self.net(Xt[idx]).squeeze(-1))
            loss = (wt[idx] * (pred - yt[idx]) ** 2).sum() / wt[idx].sum().clamp_min(1e-8)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss)

    @torch.no_grad()
    def predict(self, X):
        return torch.sigmoid(self.net(self._norm(X)).squeeze(-1)).numpy()

    def state(self):
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict(), "X": self.X, "y": self.y, "w": self.w,
                "rms": (self.rms.mean, self.rms.var, self.rms.count), "rng": self.rng.bit_generator.state}

    def load(self, s):
        self.net.load_state_dict(s["net"]); self.opt.load_state_dict(s["opt"])
        self.X, self.y, self.w = s["X"], s["y"], s["w"]
        self.rms.mean, self.rms.var, self.rms.count = s["rms"]; self.rng.bit_generator.state = s["rng"]


def evaluate(env_id, agent, obs_rms, episodes, seed0):
    """Deterministic policy (actor mean), frozen observation statistics, environment's own return."""
    env = make_env(env_id, seed0)
    rets, lens, vxs, still, final_d = [], [], [], [], []
    for k in range(episodes):
        obs, info = env.reset(seed=seed0 + k)
        done, ret, n, ep_vx, ep_still = False, 0.0, 0, [], 0
        while not done:
            x = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
            with torch.no_grad():
                a = agent.actor_mean(torch.tensor(x, dtype=torch.float32)[None]).numpy()[0]
            obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
            done = term or trunc
            ret += float(r); n += 1
            t = info["tel"]; ep_vx.append(t[0]); ep_still += int(abs(t[0]) < 0.1)
        rets.append(ret); lens.append(n); vxs.append(float(np.mean(ep_vx))); still.append(ep_still / n)
        if env_id == "Reacher-v5": final_d.append(float(info["tel"][3]))
    env.close()
    out = {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)), "returns": rets, "len_mean": float(np.mean(lens)),
           "vx_mean": float(np.mean(vxs)), "still_frac": float(np.mean(still))}
    if final_d: out["final_dist_mean"] = float(np.mean(final_d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", required=True)
    ap.add_argument("--reward", required=True, choices=REWARDS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total_timesteps", type=int, default=1_000_000)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--num_steps", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--num_minibatches", type=int, default=32)
    ap.add_argument("--update_epochs", type=int, default=10)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=100_000)
    ap.add_argument("--eval_episodes", type=int, default=10)
    ap.add_argument("--final_eval_episodes", type=int, default=20)
    ap.add_argument("--distill_frac", type=float, default=0.1)
    ap.add_argument("--log_window_frac", type=float, default=0.02)
    ap.add_argument("--port", type=int, default=6100)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "final.json").exists():
        print("already finished:", out); return
    # A checkpoint is written after every periodic evaluation; a restarted job resumes from it, with the log cut back
    # to what had been written at that point.
    ck_path = out / "ckpt.pt"
    ck = torch.load(ck_path, weights_only=False) if ck_path.exists() else None
    if ck is not None:
        log = open(out / "log.jsonl", "r+"); log.truncate(ck["log_bytes"]); log.seek(ck["log_bytes"])
    else:
        log = open(out / "log.jsonl", "w")
    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush()
    if ck is None:
        emit({"type": "config", **vars(a), "window": WINDOW[a.env_id], "torch": torch.__version__})

    torch.set_num_threads(1)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed + 7)
    env = make_env(a.env_id, a.seed)
    dt = float(env.unwrapped.dt)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    lo, hi = env.action_space.low, env.action_space.high
    H = WINDOW[a.env_id]

    agent = Agent(obs_dim, act_dim)
    optimizer = optim.Adam(agent.parameters(), lr=a.learning_rate, eps=1e-5)
    obs_rms, ret_rms, disc_ret = RunningMeanStd((obs_dim,)), RunningMeanStd(()), 0.0
    judge = None
    if a.reward.startswith(("kev", "ar")):
        from dcjev.judges.client import JudgeClient
        judge = JudgeClient(a.port)
    distiller = Distiller(len(window_features(a.env_id, np.zeros((H, 5)), False)), a.seed) if a.reward == "kev_distill" else None

    T = a.num_steps
    mb = T // a.num_minibatches
    num_updates = a.total_timesteps // T
    obs_b = torch.zeros((T, obs_dim)); act_b = torch.zeros((T, act_dim)); logp_b = torch.zeros(T); val_b = torch.zeros(T)
    done_b = np.zeros(T); term_b = np.zeros(T, dtype=bool); envr_b = np.zeros(T); tel_b = np.zeros((T, 5))

    def norm_obs(o, update=True):
        if update: obs_rms.update(o[None])
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)

    ep_ret, ep_len, global_step, t_start = 0.0, 0, 0, time.time()
    judge_calls = judge_windows = 0
    next_eval, first_update, resumes = a.eval_every, 1, 0
    if ck is None:
        raw_obs, _ = env.reset(seed=a.seed)
        next_done = 0.0
    else:
        agent.load_state_dict(ck["agent"]); optimizer.load_state_dict(ck["opt"])
        obs_rms.mean, obs_rms.var, obs_rms.count = ck["obs_rms"]; ret_rms.mean, ret_rms.var, ret_rms.count = ck["ret_rms"]
        disc_ret = ck["disc_ret"]
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        rng.bit_generator.state = ck["rng"]
        if distiller is not None: distiller.load(ck["distiller"])
        first_update, global_step, next_eval = ck["update"] + 1, ck["global_step"], ck["next_eval"]
        judge_calls, judge_windows, resumes = ck["judge_calls"], ck["judge_windows"], ck["resumes"] + 1
        if judge is not None: judge.seconds = ck["judge_s"]
        t_start = time.time() - ck["wall_s"]
        # the simulator state is not saved: the episode that was running at the checkpoint is dropped and a new one
        # starts from a reset with a seed of its own
        raw_obs, _ = env.reset(seed=a.seed + 1_000_003 * resumes)
        next_done = 1.0
        emit({"type": "resume", "step": global_step, "update": ck["update"], "resumes": resumes,
              "dropped_episode_steps": ck["ep_len"]})
    next_obs = torch.tensor(norm_obs(raw_obs), dtype=torch.float32)

    def save_ckpt(update):
        tmp = out / "ckpt.tmp"
        torch.save({"agent": agent.state_dict(), "opt": optimizer.state_dict(), "obs_rms": (obs_rms.mean, obs_rms.var, obs_rms.count),
                    "ret_rms": (ret_rms.mean, ret_rms.var, ret_rms.count), "disc_ret": disc_ret, "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "rng": rng.bit_generator.state,
                    "distiller": distiller.state() if distiller is not None else None, "update": update, "global_step": global_step,
                    "next_eval": next_eval, "judge_calls": judge_calls, "judge_windows": judge_windows, "resumes": resumes,
                    "judge_s": judge.seconds if judge is not None else 0.0, "wall_s": time.time() - t_start, "ep_len": ep_len,
                    "log_bytes": log.tell()}, tmp)
        os.replace(tmp, ck_path)

    for update in range(first_update, num_updates + 1):
        if True:  # anneal_lr
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / num_updates) * a.learning_rate
        for t in range(T):
            global_step += 1
            obs_b[t] = next_obs; done_b[t] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs[None])
            val_b[t] = value.flatten()[0]; act_b[t] = action[0]; logp_b[t] = logp[0]
            raw_obs, r, term, trunc, info = env.step(np.clip(action[0].numpy(), lo, hi))
            envr_b[t] = r; tel_b[t] = info["tel"]; term_b[t] = term
            ep_ret += float(r); ep_len += 1
            if term or trunc:
                emit({"type": "train_episode", "step": global_step, "return": ep_ret, "len": ep_len})
                ep_ret, ep_len = 0.0, 0
                raw_obs, _ = env.reset()
            next_obs = torch.tensor(norm_obs(raw_obs), dtype=torch.float32)
            next_done = float(term or trunc)

        # judge windows: consecutive steps of one episode, at most H long, cut at episode ends and at the rollout end
        ends = np.zeros(T, dtype=bool)   # step t ended its episode (t's transition was terminal or truncated)
        ends[:-1] = done_b[1:] > 0; ends[-1] = next_done > 0
        windows, s = [], 0
        for t in range(T):
            if t - s + 1 == H or ends[t] or t == T - 1:
                windows.append((s, t + 1)); s = t + 1
        rew = np.zeros(T)
        w_scores = None
        if a.reward == "env":
            rew[:] = envr_b
        elif a.reward == "env_window":
            for (s0, s1) in windows: rew[s0:s1] = envr_b[s0:s1].mean()
        else:
            summaries = [window_summary(a.env_id, tel_b[s0:s1], bool(term_b[s1 - 1]), dt) for (s0, s1) in windows]
            if a.reward in ("kev_score", "kev_gated"):
                qs = ("progress",) if a.reward == "kev_score" else ("progress", "healthy")
                res = judge.score("kev", a.env_id, summaries, qs)
                p = np.array(res["progress"]); w_scores = p if a.reward == "kev_score" else p * np.array(res["healthy"])
                judge_calls += 1; judge_windows += len(summaries)
            elif a.reward in ("ar_gated", "ar_sampled"):
                res = judge.score("ar", a.env_id, summaries, ("progress", "healthy"))
                p = np.array(res["progress"] if a.reward == "ar_gated" else res["progress_sampled"])
                w_scores = p * np.array(res["healthy"])
                judge_calls += 1; judge_windows += len(summaries)
            else:   # kev_distill
                feats = np.stack([window_features(a.env_id, tel_b[s0:s1], bool(term_b[s1 - 1])) for (s0, s1) in windows])
                k = len(windows) if update == 1 else max(4, int(round(a.distill_frac * len(windows))))
                lab = rng.choice(len(windows), size=min(k, len(windows)), replace=False)
                res = judge.score("kev", a.env_id, [summaries[i] for i in lab], ("progress", "healthy"))
                y = np.array(res["progress"]) * np.array(res["healthy"]); conf = np.array(res["progress_conf"])
                distiller.add(feats[lab], y, np.clip(conf, 0.05, 1.0))
                dloss = distiller.train()
                w_scores = distiller.predict(feats); w_scores[lab] = y
                judge_calls += 1; judge_windows += len(lab)
            for (s0, s1), sc in zip(windows, w_scores): rew[s0:s1] = sc
        # a sample of windows for the judge-agreement analysis (every condition logs them)
        for i in np.flatnonzero(rng.random(len(windows)) < a.log_window_frac):
            s0, s1 = windows[i]
            rec = {"type": "window", "step": global_step - T + s1, "summary": window_summary(a.env_id, tel_b[s0:s1], bool(term_b[s1 - 1]), dt),
                   "env_reward_sum": float(envr_b[s0:s1].sum()), "len": int(s1 - s0)}
            if w_scores is not None: rec["train_reward"] = float(w_scores[i])
            emit(rec)
        # reward normalisation exactly as gymnasium's NormalizeReward (+ TransformReward clip to [-10, 10])
        rew_n = np.zeros(T)
        for t in range(T):
            disc_ret = disc_ret * a.gamma * (1 - float(term_b[t])) + rew[t]
            ret_rms.update(np.array([disc_ret]))
            rew_n[t] = np.clip(rew[t] / np.sqrt(ret_rms.var + 1e-8), -10, 10)
        rewards = torch.tensor(rew_n, dtype=torch.float32)

        # GAE (CleanRL)
        with torch.no_grad():
            next_value = agent.get_value(next_obs[None]).flatten()[0]
            adv = torch.zeros(T); lastgaelam = 0.0
            dones = torch.tensor(done_b, dtype=torch.float32)
            for t in reversed(range(T)):
                if t == T - 1:
                    nonterminal, nextvalues = 1.0 - next_done, next_value
                else:
                    nonterminal, nextvalues = 1.0 - dones[t + 1], val_b[t + 1]
                delta = rewards[t] + a.gamma * nextvalues * nonterminal - val_b[t]
                adv[t] = lastgaelam = delta + a.gamma * a.gae_lambda * nonterminal * lastgaelam
            returns = adv + val_b

        # PPO update (CleanRL)
        inds = np.arange(T); clipfracs = []
        for epoch in range(a.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, T, mb):
                mbi = inds[start:start + mb]
                _, newlogp, entropy, newv = agent.get_action_and_value(obs_b[mbi], act_b[mbi])
                logratio = newlogp - logp_b[mbi]; ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > a.clip_coef).float().mean().item())
                madv = adv[mbi]; madv = (madv - madv.mean()) / (madv.std() + 1e-8)
                pg_loss = torch.max(-madv * ratio, -madv * torch.clamp(ratio, 1 - a.clip_coef, 1 + a.clip_coef)).mean()
                newv = newv.view(-1)
                v_unclipped = (newv - returns[mbi]) ** 2
                v_clipped = val_b[mbi] + torch.clamp(newv - val_b[mbi], -a.clip_coef, a.clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - returns[mbi]) ** 2).mean()
                loss = pg_loss - a.ent_coef * entropy.mean() + v_loss * a.vf_coef
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), a.max_grad_norm); optimizer.step()

        rec = {"type": "update", "step": global_step, "sps": global_step / (time.time() - t_start), "pg_loss": float(pg_loss),
               "v_loss": float(v_loss), "approx_kl": float(approx_kl), "clipfrac": float(np.mean(clipfracs)),
               "raw_reward_mean": float(rew.mean()), "env_reward_mean": float(envr_b.mean()), "windows": len(windows)}
        if w_scores is not None:
            rec["corr_window_score_env"] = float(np.corrcoef(w_scores, [envr_b[s0:s1].mean() for s0, s1 in windows])[0, 1]) if len(windows) > 2 and np.std(w_scores) > 0 else None
        if judge is not None:
            rec["judge_wait_s"] = judge.seconds; rec["judge_windows"] = judge_windows
        emit(rec)
        if global_step >= next_eval or update == num_updates:
            ev = evaluate(a.env_id, agent, copy.deepcopy(obs_rms), a.eval_episodes, 10_000)
            emit({"type": "eval", "step": global_step, **ev}); next_eval += a.eval_every
            if update < num_updates: save_ckpt(update)

    final = evaluate(a.env_id, agent, copy.deepcopy(obs_rms), a.final_eval_episodes, 20_000)
    torch.save({"agent": agent.state_dict(), "obs_mean": obs_rms.mean, "obs_var": obs_rms.var}, out / "policy.pt")
    summary = {"env_id": a.env_id, "reward": a.reward, "seed": a.seed, "steps": global_step, "wall_s": time.time() - t_start,
               "judge_windows": judge_windows, "judge_calls": judge_calls,
               "judge_wait_s": judge.seconds if judge else 0.0, "resumes": resumes, **{f"final_{k}": v for k, v in final.items()}}
    emit({"type": "final", **summary})
    (out / "final.json").write_text(json.dumps(summary, indent=1))
    log.close()
    ck_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
