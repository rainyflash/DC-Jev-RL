"""GRPO for Part III-A: multi-turn code repair with execution feedback (dcjev.code.episode), Qwen3.5-4B-Base + LoRA with
Kev-4B on the same backbone (dcjev.llm.shared), rollouts from the vLLM server, the same update as Part II
(dcjev.llm.grpo.policy_update: group baseline, batch scale, token-level loss, TIS, one optimizer step per batch).

    python -m dcjev.code.grpo_code --cond vis --seed 1 --out /content/work/runs/part3a/vis/s1

Reward of an episode, from its final attempt:
  vis      1 if it passes every visible test (all an agent holding only the visible tests can check), else 0
  vis_kev  vis x Kev P(yes) for 'genuine' (does the code solve the problem for all valid inputs instead of special-casing
           the example tests; experiments/part3a_code/prompts/code_prompts.json)
  hidden   1 if it passes every visible and every held-out test (an oracle: what vis would be with perfect tests)
Every attempt of an episode is trained with the episode's advantage (the feedback tokens are context, not trained).
Logged for every condition: held-out pass, hack rate (visible passed, held-out failed), attempts, and Kev's 'genuine'
and 'correct' answers on the final code (every episode in vis_kev, a random --kev_log_n per step otherwise).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from dcjev.code.data import load
from dcjev.code.episode import config, kev_code_request, run_episodes, summary
from dcjev.code.sandbox import Executor
from dcjev.llm.grpo import Rollouts, Sampler, load_ckpt, policy_update, publish, save_ckpt
from dcjev.llm.prompts import kev_scalars
from dcjev.metrics import auroc

CONDS = ["vis", "vis_kev", "hidden"]


def evaluate(ro, sb, lora, sets, ex, judge_sets=(), dump=None):
    """Greedy episodes (one per problem) on each set: summary rates, and Kev's view of the final code on judge_sets."""
    res = {}
    for name, recs in sets.items():
        t0 = time.time()
        eps = run_episodes(ro, sb.tok, recs, 1, lora, ex, temperature=0.0)
        e = {**summary(eps), "seconds": round(time.time() - t0, 1)}
        ks = None
        if name in judge_sets:
            probs, metas = sb.kev_probs([kev_code_request(x["rec"], x["code"]) for x in eps])
            ks = [kev_scalars(p, m) for p, m in zip(probs, metas)]
            allp = np.array([x["vis_pass"] and x["held_pass"] for x in eps]); vis = np.array([x["vis_pass"] for x in eps])
            g = np.array([k["genuine"] for k in ks])
            e["kev"] = {q: float(np.mean([k[q] for k in ks])) for q in ("genuine", "correct")}
            e["kev_genuine_auroc_all_pass_given_vis"] = auroc(g[vis], allp[vis]) if vis.any() else None
        if dump is not None:
            for n, x in enumerate(eps):
                dump.write(json.dumps({"set": name, "id": x["rec"]["id"], "attempts": [t["c"]["text"] for t in x["turns"]],
                                       "vis_pass": x["vis_pass"], "held_pass": x["held_pass"], "kev": ks[n] if ks else None}) + "\n")
        res[name] = e
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/content/work/data/code")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tis_cap", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--micro_tokens", type=int, default=16384)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--periodic", default="kodcode_val,mbppplus,humanevalplus")
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--rollout_port", type=int, default=6200)
    ap.add_argument("--final_eval", type=int, default=1)
    ap.add_argument("--kev_log_n", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "final.json").exists():
        print("already finished:", out); return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from dcjev.llm.shared import SharedBackbone
    t0 = time.time()
    sb = SharedBackbone(lora_r=a.lora_r, lora_alpha=2 * a.lora_r)
    params = sb.policy_parameters()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    train = load(f"{a.data}/kodcode_train.jsonl")
    sampler = Sampler(train, a.seed)
    sets = {k: load(f"{a.data}/{k}.jsonl") for k in ("kodcode_val", "mbppplus", "humanevalplus")}
    periodic = {k: v for k, v in sets.items() if k in a.periodic.split(",")}
    ex = Executor(a.workers)
    ro = Rollouts(a.rollout_port)
    tag = f"code-{a.cond}-s{a.seed}-{zlib.crc32(str(out.resolve()).encode()) & 0xffffff:06x}"
    load_s = time.time() - t0

    start = 0
    if (out / "ckpt" / "state.pt").exists():
        start = load_ckpt(out, sb, opt, sampler)
        keep = [l for l in open(out / "log.jsonl") if json.loads(l).get("step", -1) < start or json.loads(l)["kind"] == "config"]
        open(out / "log.jsonl", "w").writelines(keep)
        if (out / "episodes.jsonl.gz").exists():
            kept = []
            try:
                with gzip.open(out / "episodes.jsonl.gz", "rt") as f:
                    for l in f:
                        if json.loads(l)["step"] < start: kept.append(l)
            except (EOFError, gzip.BadGzipFile, json.JSONDecodeError):
                pass
            with gzip.open(out / "episodes.jsonl.gz", "wt") as f: f.writelines(kept)
        print(f"resumed at step {start}", flush=True)
    log = open(out / "log.jsonl", "a")
    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush()
    if start == 0:
        emit({"kind": "config", **vars(a), "train_size": len(train), **{f"{k}_size": len(v) for k, v in sets.items()},
              "prompts_version": config()["version"], "load_seconds": round(load_s, 1),
              "mem_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2)})
    roll = gzip.open(out / "episodes.jsonl.gz", "at")

    for step in range(start, a.steps + 1):
        lora = publish(sb, out, step, tag)
        if step % a.eval_every == 0 or step == a.steps:
            emit({"kind": "eval", "step": step, **evaluate(ro, sb, lora, periodic, ex)})
        if step == a.steps:
            break
        torch.cuda.reset_peak_memory_stats()
        tm = {}
        batch = sampler.next(a.prompts)
        t = time.time()
        eps = run_episodes(ro, sb.tok, batch, a.group, lora, ex, temperature=a.temperature,
                           seed=(a.seed * 1_000_003 + step * 10_007) % 2**31)
        tm["episodes"] = time.time() - t
        vis = np.array([e["vis_pass"] for e in eps]); held = np.array([e["held_pass"] for e in eps]); allp = vis & held
        t = time.time()
        scored = np.arange(len(eps)) if a.cond == "vis_kev" else np.sort(np.random.default_rng(step).choice(len(eps), min(a.kev_log_n, len(eps)), replace=False))
        probs, metas = sb.kev_probs([kev_code_request(eps[k]["rec"], eps[k]["code"]) for k in scored])
        ks = [None] * len(eps)
        for k, p, m in zip(scored, probs, metas): ks[k] = kev_scalars(p, m)
        tm["kev"] = time.time() - t
        if a.cond == "vis": R = vis.astype(float)
        elif a.cond == "hidden": R = allp.astype(float)
        else: R = vis * np.array([k["genuine"] for k in ks])
        P, G = len(batch), a.group
        Cn = (R.reshape(P, G) - R.reshape(P, G).mean(1, keepdims=True)).reshape(-1)
        A = Cn / (Cn.std() + 1e-4)
        t = time.time()
        seqs, adv = [], []
        for e, x in zip(eps, A):
            for tr in e["turns"]:
                seqs.append((tr["ctx"], tr["c"]["token_ids"], tr["c"]["logprobs"])); adv.append(x)
        n_tok = sum(len(s[1]) for s in seqs)
        upd = policy_update(sb, opt, params, seqs, adv, n_tok, a)
        tm["train"] = time.time() - t
        g = np.array([ks[k]["genuine"] for k in scored]); vs = vis[scored]
        rec = {"kind": "step", "step": step, "epoch": sampler.epoch, "reward_mean": float(R.mean()), **summary(eps),
               "groups_all_vis_fail": float((vis.reshape(P, G).max(1) == 0).mean()), "groups_all_vis_pass": float((vis.reshape(P, G).min(1) == 1).mean()),
               "groups_zero_reward_var": float((R.reshape(P, G).std(1) == 0).mean()), "adv_scale": float(Cn.std()),
               "kev": {q: float(np.mean([ks[k][q] for k in scored])) for q in ("genuine", "correct")}, "kev_scored": len(scored),
               "kev_genuine_auroc_all_pass_given_vis": auroc(g[vs], allp[scored][vs]) if vs.any() else None,
               "kev_correct_auroc_all_pass": auroc([ks[k]["correct"] for k in scored], allp[scored]),
               "attempt_seqs": len(seqs), "tokens": int(n_tok), **upd,
               "mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "seconds": {k: round(x, 2) for k, x in tm.items()}}
        emit(rec)
        for k, e in enumerate(eps):
            roll.write(json.dumps({"step": step, "id": e["rec"]["id"], "k": k % G, "attempts": [t["c"]["text"] for t in e["turns"]],
                                   "vis_pass": bool(e["vis_pass"]), "held_pass": bool(e["held_pass"]), "kev": ks[k],
                                   "r": float(R[k]), "adv": float(A[k])}) + "\n")
        roll.flush()
        print(f"step {step} vis {vis.mean():.3f} all {allp.mean():.3f} hack {rec['hack']:.3f} att {rec['attempts_mean']:.2f} "
              f"R {R.mean():.3f} gn {upd['grad_norm']:.3f} t {rec['seconds']}", flush=True)
        if (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            save_ckpt(out, sb, opt, sampler, step + 1)

    final = {"config": vars(a)}
    if a.final_eval:
        with gzip.open(out / "final_episodes.jsonl.gz", "wt") as dump:
            final["eval"] = evaluate(ro, sb, lora, sets, ex, judge_sets=tuple(sets), dump=dump)
        emit({"kind": "final", "step": a.steps, **final["eval"]})
    sb.save_policy(out / "policy_final")
    (out / "final.json").write_text(json.dumps(final, indent=1))
    roll.close()
    for d in ("lora_hf", "lora_vllm_0", "lora_vllm_1", "ckpt"):
        if (out / d).exists(): shutil.rmtree(out / d)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
