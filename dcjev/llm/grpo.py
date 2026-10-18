"""GRPO for Part II: Qwen3.5-4B-Base with a LoRA policy; rewards from the math verifier, optionally combined with a judge.

    python -m dcjev.llm.grpo --cond rlvr --seed 1 --out /content/work/runs/part2/rlvr/s1

The trainer (kev's environment) holds the policy and Kev-4B on one backbone (dcjev.llm.shared). Rollouts come from a vLLM
server in its own environment (dcjev.llm.rollout_server), which receives the policy LoRA after every update; the remaining
train/inference mismatch is corrected with truncated importance sampling (TIS; Yao et al., 2025).

Reward of a completion (v = verifier 0/1; Pc, Pg = Kev's P(yes) for the 'correct' and 'genuine' questions of
experiments/part2_llm/prompts/judge_prompts.json; Pa = the autoregressive judge's p(yes) for 'correct'):
  rlvr           v
  rlvr_kev_add   v + lam * Pc
  rlvr_kev_gate  v * Pg
  kev_only       Pc                  (no verifier: is the judge alone exploitable?)
  rlvr_ar_add    v + lam * Pa
With lam < 1 a correct completion always outranks an incorrect one in the additive conditions.

Label-budget conditions (Part III-B): only --label_frac of each group's completions get the verifier's label (chosen at
random); the others are
  rlvr_sub       left out of the update (advantages from the labelled completions only)
  sub_kev        rewarded with Kev's P(correct), frozen as released
  sub_kev_recal  rewarded with Kev's P(correct) after its pointer head is refitted online on all labelled completions so
                 far (cross-entropy to the verifier's label, anchored to the released head; dcjev.llm.grpo.HeadRecal)
  sub_probe      rewarded by a head of the same shape trained from scratch on the same labels and Kev features
The verifier's label of every unlabelled completion is still computed, for logging only (how good the fill-in reward was).

Tree conditions (Part III-C; dcjev.llm.tree): reward v, but each group's leaves grow from --roots sampled prefixes of
--prefix_tokens tokens, --keep of which are continued, chosen by Kev's 'on_track' answer (tree_kev) or at random (tree_random).

Advantage: the reward minus its group's mean, divided by the standard deviation of these centred rewards over the batch
(group baseline, batch scale: Liu et al. 2025 "Lite PPO"; TRL's scale_rewards="batch"), so lam keeps its meaning in groups
where the verifier gives every completion the same reward. Loss: token-level policy gradient (summed over the completion
tokens of the batch, divided by their number), one optimizer step per batch (on-policy: PPO's ratio is 1 and its clipping
is inactive), each token weighted by min(pi_train / pi_vllm, tis_cap); no KL term.
Kev's three answers are computed for every completion in every condition (logged; used in the reward only as above).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import time
import zlib
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import torch

from dcjev.llm.data import PROMPT, STOP, load, train_val
from dcjev.llm.lora_io import to_vllm
from dcjev.llm.prompts import QUESTIONS, ar_messages, clip_solution, kev_request, kev_scalars
from dcjev.llm.rollout_server import AUTHKEY
from dcjev.llm.tree import tree_log, tree_rollout
from dcjev.llm.verify import Verifier
from dcjev.metrics import auroc, brier, ece, within_group_auroc

CONDS = ["rlvr", "rlvr_kev_add", "rlvr_kev_gate", "kev_only", "rlvr_ar_add", "rlvr_sub", "sub_kev", "sub_kev_recal", "sub_probe",
         "tree_kev", "tree_random"]
SUB = {"rlvr_sub", "sub_kev", "sub_kev_recal", "sub_probe"}
TREE = {"tree_kev", "tree_random"}
KEV_REWARD = {"rlvr_kev_add", "rlvr_kev_gate", "kev_only", "sub_kev", "sub_kev_recal", "sub_probe"}   # Kev scores every completion


class HeadRecal:
    """A copy of Kev's pointer head refitted online on (Kev features, verifier label) pairs for the 'correct' question.

    The released head on the backbone is left untouched; this copy scores the unlabelled completions. Loss: binary
    cross-entropy of P(yes) = softmax(z / T)[yes] against the label, plus anchor * ||theta - theta_released||^2 (none for
    the from-scratch probe, which also starts from a fresh initialisation and T = 1)."""
    def __init__(self, head, temperature, scratch, lr=1e-4, anchor=1e-3, buffer=8192, steps=50, batch=256, seed=0):
        import copy
        self.head = copy.deepcopy(head).float().train()
        for p in self.head.parameters(): p.requires_grad_(True)
        if scratch:
            torch.manual_seed(seed)
            for m in (self.head.q, self.head.k): m.reset_parameters()
        self.T = 1.0 if scratch else float(temperature)
        self.anchor = 0.0 if scratch else anchor
        self.theta0 = [p.detach().clone() for p in self.head.parameters()]
        self.opt = torch.optim.Adam(self.head.parameters(), lr=lr)
        self.buffer, self.steps, self.batch = buffer, steps, batch
        self.hd, self.ho, self.y = [], [], []
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def logits(self, hd, ho):   # [B, d], [B, 2, d] -> yes-minus-no logit [B]
        z = (self.head.k(ho) * self.head.q(hd)[:, None, :]).sum(-1) * self.head.scale / self.T
        return z[:, 1] - z[:, 0]

    def add(self, feats, labels):
        for (hd, ho), y in zip(feats, labels):
            self.hd.append(hd); self.ho.append(ho); self.y.append(float(y))
        self.hd, self.ho, self.y = self.hd[-self.buffer:], self.ho[-self.buffer:], self.y[-self.buffer:]

    def fit(self):
        import torch.nn.functional as F
        Hd, Ho, Y = torch.stack(self.hd), torch.stack(self.ho), torch.tensor(self.y, device=self.hd[0].device)
        tot = 0.0
        for _ in range(self.steps):
            i = torch.randint(len(Y), (min(self.batch, len(Y)),), generator=self.gen).to(Y.device)
            loss = F.binary_cross_entropy_with_logits(self.logits(Hd[i], Ho[i]), Y[i])
            if self.anchor:
                loss = loss + self.anchor * sum(((p - p0) ** 2).sum() for p, p0 in zip(self.head.parameters(), self.theta0))
            self.opt.zero_grad(); loss.backward(); self.opt.step(); tot += loss.item()
        return tot / self.steps

    @torch.no_grad()
    def p_yes(self, feats):
        if not feats: return np.zeros(0)
        return torch.sigmoid(self.logits(torch.stack([f[0] for f in feats]), torch.stack([f[1] for f in feats]))).cpu().numpy()


class Rollouts:
    def __init__(self, port, retries=720):
        for k in range(retries):
            try:
                self.conn = Client(("127.0.0.1", port), authkey=AUTHKEY)
                break
            except ConnectionRefusedError:
                if k == retries - 1:
                    raise
                time.sleep(5)

    def __call__(self, prompts=None, **kw):
        """prompts: strings, or prompt_ids=[[int]] (token ids continued as given) instead."""
        self.conn.send({"op": "generate", "prompts": prompts, **kw})
        r = self.conn.recv()
        if "error" in r:
            raise RuntimeError("rollout server error:\n" + r["error"])
        return r


class Sampler:
    """Prompts in epochs of seeded permutations."""
    def __init__(self, recs, seed):
        self.recs, self.rng = recs, np.random.default_rng(seed)
        self.perm, self.pos, self.epoch = self.rng.permutation(len(recs)), 0, 0

    def next(self, k):
        out = []
        while len(out) < k:
            if self.pos == len(self.perm):
                self.perm, self.pos, self.epoch = self.rng.permutation(len(self.recs)), 0, self.epoch + 1
            out.append(self.recs[self.perm[self.pos]]); self.pos += 1
        return out

    def state(self):
        return {"perm": self.perm.tolist(), "pos": self.pos, "epoch": self.epoch, "rng": self.rng.bit_generator.state}

    def load(self, s):
        self.perm, self.pos, self.epoch = np.array(s["perm"]), s["pos"], s["epoch"]
        self.rng.bit_generator.state = s["rng"]


def reward(cond, v, ks, pa, lam):
    if cond == "rlvr" or cond in TREE: return float(v)
    if cond == "rlvr_kev_add": return v + lam * ks["correct"]
    if cond == "rlvr_kev_gate": return v * ks["genuine"]
    if cond == "kev_only": return ks["correct"]
    if cond == "rlvr_ar_add": return v + lam * pa
    raise ValueError(cond)


def pack(idx, lengths, budget):
    """Micro-batches of sequence indices (sorted by length) whose padded size stays within `budget` tokens."""
    idx = sorted(idx, key=lambda k: lengths[k])
    out, cur = [], []
    for k in idx:
        if cur and (len(cur) + 1) * lengths[k] > budget:
            out.append(cur); cur = []
        cur.append(k)
    if cur: out.append(cur)
    return out


def policy_update(sb, opt, params, seqs, adv, n_tok, a):
    """One optimizer step on seqs [(prompt ids, completion ids, vLLM log-probabilities of the completion)] with one
    advantage per sequence: the token-level policy-gradient loss summed over the completion tokens and divided by n_tok,
    each token weighted by min(pi_train / pi_vllm, tis_cap). Sequences with zero advantage contribute no gradient and
    are skipped."""
    sb.model.train()
    lengths = [len(p) + len(c) for p, c, _ in seqs]
    trained = [k for k in range(len(seqs)) if adv[k] != 0 and len(seqs[k][1]) > 0]
    loss_sum, w_sum, w_clip, d_sum, d_n = 0.0, 0.0, 0, 0.0, 0
    for mb in pack(trained, lengths, a.micro_tokens):
        lps = sb.token_logprobs([(seqs[k][0], seqs[k][1]) for k in mb])
        loss = 0.0
        for k, lp in zip(mb, lps):
            lv = torch.tensor(seqs[k][2], device=lp.device, dtype=torch.float32)
            d = lp.detach() - lv
            w = torch.exp(d).clamp(max=a.tis_cap)
            loss = loss - (w * float(adv[k]) * lp).sum()
            w_sum += float(w.sum()); w_clip += int((torch.exp(d) > a.tis_cap).sum()); d_sum += float(d.abs().sum()); d_n += len(lv)
        (loss / n_tok).backward()
        loss_sum += float(loss.detach()) / n_tok
    gn = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip))
    opt.step(); opt.zero_grad(set_to_none=True)
    return {"trained_seqs": len(trained), "loss": loss_sum, "grad_norm": gn, "tis_w_mean": w_sum / max(d_n, 1),
            "tis_clip_frac": w_clip / max(d_n, 1), "lp_absdiff_mean": d_sum / max(d_n, 1)}


def publish(sb, out, step, tag):
    """Save the policy LoRA, rename it for vLLM and return the request entry (a new name per update)."""
    src = out / "lora_hf"
    if src.exists(): shutil.rmtree(src)
    sb.save_policy(src)
    dst = to_vllm(src, out / f"lora_vllm_{step % 2}")
    return {"name": f"{tag}-{step}", "path": str(dst)}


def flat(g):
    return [c for o in g["outputs"] for c in o]


def evaluate(ro, lora, sets, verifier, max_tokens, sb=None, judge_sets=(), dump=None):
    """sets: name -> (records, samples per problem, temperature, top_p). Greedy when temperature == 0.
    For the sets in judge_sets the completions are also scored by Kev (on the shared backbone), to compare Kev's view of
    the policy with the verifier's after training on each reward."""
    res = {}
    for name, (recs, n, temp, top_p) in sets.items():
        t0 = time.time()
        g = ro([PROMPT.format(problem=r["problem"]) for r in recs], n=n, temperature=temp, top_p=top_p, max_tokens=max_tokens,
               stop=STOP, seeds=[1000 * i for i in range(len(recs))] if temp > 0 else None, lora=lora)
        cs = flat(g)
        golds = [r["answer"] for r in recs for _ in range(n)]
        v = np.array(verifier([c["text"] for c in cs], golds)).reshape(len(recs), n)
        L = np.array([len(c["token_ids"]) for c in cs])
        e = {"acc": float(v.mean()), "n_problems": len(recs), "samples": n, "temperature": temp, "top_p": top_p,
             "len_mean": float(L.mean()), "truncated": float(np.mean([c["finish_reason"] == "length" for c in cs])),
             "seconds": round(time.time() - t0, 1)}
        if n > 1: e["pass_at_n"] = float(v.max(1).mean())
        vv, ks = v.reshape(-1), None
        if name in judge_sets and sb is not None:
            probs, metas = sb.kev_probs([kev_request(recs[i // n]["problem"], clip_solution(sb.tok, c["text"])) for i, c in enumerate(cs)])
            ks = [kev_scalars(p, m) for p, m in zip(probs, metas)]
            pc = np.array([k["correct"] for k in ks])
            e["kev"] = {q: float(np.mean([k[q] for k in ks])) for q in QUESTIONS}
            e["kev_correct_auroc"], e["kev_correct_ece"], e["kev_correct_brier"] = auroc(pc, vv), ece(pc, vv), brier(pc, vv)
        if dump is not None:
            for i, c in enumerate(cs):
                dump.write(json.dumps({"set": name, "id": recs[i // n]["id"], "k": i % n, "text": c["text"], "n_tok": len(c["token_ids"]),
                                       "finish": c["finish_reason"], "v": int(vv[i]), "kev": ks[i] if ks else None}) + "\n")
        res[name] = e
    return res


def save_ckpt(out, sb, opt, sampler, step, recal=None):
    tmp, dst = out / "ckpt_tmp", out / "ckpt"
    if tmp.exists(): shutil.rmtree(tmp)
    sb.save_policy(tmp / "policy")
    torch.save({"opt": opt.state_dict(), "sampler": sampler.state(), "step": step, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()}, tmp / "state.pt")
    if recal is not None:
        torch.save({"head": recal.head.state_dict(), "opt": recal.opt.state_dict(), "y": recal.y, "gen": recal.gen.get_state(),
                    "hd": torch.stack(recal.hd).cpu() if recal.hd else None, "ho": torch.stack(recal.ho).cpu() if recal.ho else None}, tmp / "recal.pt")
    if dst.exists(): shutil.rmtree(dst)
    tmp.rename(dst)


def load_ckpt(out, sb, opt, sampler, recal=None):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    ck = out / "ckpt"
    w = load_file(str(ck / "policy" / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(sb.model, w, adapter_name="default")
    if res.unexpected_keys: raise ValueError(f"checkpoint tensors not placed: {res.unexpected_keys[:3]}")
    st = torch.load(ck / "state.pt", weights_only=False)
    opt.load_state_dict(st["opt"]); sampler.load(st["sampler"])
    torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state(st["cuda_rng"])
    if recal is not None:
        r = torch.load(ck / "recal.pt", weights_only=False)
        recal.head.load_state_dict(r["head"]); recal.opt.load_state_dict(r["opt"]); recal.gen.set_state(r["gen"]); recal.y = r["y"]
        dev = next(recal.head.parameters()).device
        recal.hd = list(r["hd"].to(dev)) if r["hd"] is not None else []
        recal.ho = list(r["ho"].to(dev)) if r["ho"] is not None else []
    return st["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/content/work/data")
    ap.add_argument("--train_sets", default="math_train")
    ap.add_argument("--math_levels", default="3,4,5", help="MATH train levels used (other sources unaffected); '' = all")
    ap.add_argument("--holdout", type=int, default=500, help="training problems held out (fixed) as the validation set used for model selection")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--tis_cap", type=float, default=2.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--eval_max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--micro_tokens", type=int, default=16384)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_gsm8k", type=int, default=500, help="size of the fixed GSM8K-test subset for periodic evaluation")
    ap.add_argument("--periodic", default="val,math500,gsm8k500", help="sets evaluated every --eval_every steps")
    ap.add_argument("--eval_limit", type=int, default=0, help="debugging: evaluate on the first N problems of each set only")
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--rollout_port", type=int, default=6200)
    ap.add_argument("--final_eval", type=int, default=1)
    ap.add_argument("--ar_model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--ar_revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--save_rollouts", type=int, default=1)
    ap.add_argument("--kev_log_n", type=int, default=64, help="conditions whose reward does not use Kev: completions per step scored by Kev for the log")
    ap.add_argument("--label_frac", type=float, default=0.25, help="label-budget conditions: fraction of each group labelled")
    ap.add_argument("--recal_lr", type=float, default=1e-4)
    ap.add_argument("--recal_anchor", type=float, default=1e-3)
    ap.add_argument("--recal_steps", type=int, default=50)
    ap.add_argument("--recal_buffer", type=int, default=8192)
    ap.add_argument("--roots", type=int, default=4, help="tree conditions: prefixes sampled per prompt")
    ap.add_argument("--keep", type=int, default=2, help="tree conditions: prefixes continued per prompt (--group / --keep samples each)")
    ap.add_argument("--prefix_tokens", type=int, default=128)
    ap.add_argument("--audit_every", type=int, default=10, help="tree conditions: steps on which the roots not kept are continued too (logged only)")
    a = ap.parse_args()
    if a.cond in TREE and (a.group % a.keep or a.keep > a.roots):
        ap.error("--group must be a multiple of --keep, and --keep at most --roots")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "final.json").exists():
        print("already finished:", out); return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from dcjev.llm.shared import SharedBackbone
    t0 = time.time()
    sb = SharedBackbone(lora_r=a.lora_r, lora_alpha=2 * a.lora_r)
    ar = None
    if a.cond == "rlvr_ar_add":
        from dcjev.judges.ar_judge import ARJudge
        ar = ARJudge(a.ar_model, revision=a.ar_revision, fused=True)
    recal = None
    if a.cond in ("sub_kev_recal", "sub_probe"):
        recal = HeadRecal(sb.head, sb.head.temperature, scratch=a.cond == "sub_probe", lr=a.recal_lr, anchor=a.recal_anchor,
                          buffer=a.recal_buffer, steps=a.recal_steps, seed=a.seed)
    params = sb.policy_parameters()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    train, val = train_val(a.data, a.train_sets, a.math_levels, a.holdout)
    sampler = Sampler(train, a.seed)
    gsm = load(f"{a.data}/gsm8k_test.jsonl")
    gsm_sub = [gsm[i] for i in sorted(np.random.default_rng(0).choice(len(gsm), a.eval_gsm8k, replace=False))]
    periodic = {"math500": (load(f"{a.data}/math500.jsonl"), 1, 0.0, 1.0), "gsm8k500": (gsm_sub, 1, 0.0, 1.0)}
    if val: periodic["val"] = (val, 1, 0.0, 1.0)
    math500 = periodic["math500"]
    periodic = {k: v for k, v in periodic.items() if k in a.periodic.split(",")}
    if a.eval_limit: periodic = {k: (v[0][:a.eval_limit],) + v[1:] for k, v in periodic.items()}
    verifier = Verifier()
    ro = Rollouts(a.rollout_port)
    tag = f"{a.cond}-s{a.seed}-{zlib.crc32(str(out.resolve()).encode()) & 0xffffff:06x}"
    load_s = time.time() - t0

    start = 0
    if (out / "ckpt" / "state.pt").exists():
        start = load_ckpt(out, sb, opt, sampler, recal)
        keep = [l for l in open(out / "log.jsonl") if json.loads(l).get("step", -1) < start or json.loads(l)["kind"] == "config"]
        open(out / "log.jsonl", "w").writelines(keep)
        if (out / "rollouts.jsonl.gz").exists():   # drop rollouts logged after the checkpoint (and a member cut by the crash)
            kept = []
            try:
                with gzip.open(out / "rollouts.jsonl.gz", "rt") as f:
                    for l in f:
                        if json.loads(l)["step"] < start: kept.append(l)
            except (EOFError, gzip.BadGzipFile, json.JSONDecodeError):
                pass
            with gzip.open(out / "rollouts.jsonl.gz", "wt") as f: f.writelines(kept)
        print(f"resumed at step {start}", flush=True)
    log = open(out / "log.jsonl", "a")
    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush()
    if start == 0:
        emit({"kind": "config", **vars(a), "train_size": len(train), "val_size": len(val), "kev_tensors": sb.kev_tensors,
              "policy_params": sum(p.numel() for p in params), "load_seconds": round(load_s, 1),
              "mem_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2)})
    roll = gzip.open(out / "rollouts.jsonl.gz", "at") if a.save_rollouts else None

    for step in range(start, a.steps + 1):
        lora = publish(sb, out, step, tag)
        if step % a.eval_every == 0 or step == a.steps:
            (out / "evals").mkdir(exist_ok=True)
            with gzip.open(out / "evals" / f"step{step}.jsonl.gz", "wt") as dump:   # per-problem results, for paired comparisons
                emit({"kind": "eval", "step": step, **evaluate(ro, lora, periodic, verifier, a.eval_max_tokens, dump=dump)})
        if step == a.steps:
            break
        torch.cuda.reset_peak_memory_stats()
        tm = {}
        batch = sampler.next(a.prompts)
        t = time.time()
        tree = None
        if a.cond in TREE:
            items, tree = tree_rollout(ro, sb, batch, a, step, lora, audit=step % a.audit_every == 0)
        else:
            g = ro([PROMPT.format(problem=r["problem"]) for r in batch], n=a.group, temperature=a.temperature, top_p=1.0,
                   max_tokens=a.max_tokens, stop=STOP, lora=lora,
                   seeds=[(a.seed * 1_000_003 + step * 10_007 + i * a.group) % 2**31 for i in range(len(batch))])
            items = [(i, g["prompt_token_ids"][i], c) for i in range(len(batch)) for c in g["outputs"][i]]
        tm["gen"] = time.time() - t
        t = time.time()
        v = np.array(verifier([c["text"] for _, _, c in items], [batch[i]["answer"] for i, _, _ in items]))
        tm["verify"] = time.time() - t
        t = time.time()
        needs_kev = a.cond in KEV_REWARD
        scored = np.arange(len(items)) if needs_kev else np.sort(np.random.default_rng(step).choice(len(items), min(a.kev_log_n, len(items)), replace=False))
        reqs = [kev_request(batch[items[k][0]]["problem"], clip_solution(sb.tok, items[k][2]["text"])) for k in scored]
        if recal is not None:
            probs, metas, feats = sb.kev_probs_features(reqs, question=QUESTIONS.index("correct"))
        else:
            probs, metas = sb.kev_probs(reqs)
        ks = [None] * len(items)
        for k, p, m in zip(scored, probs, metas): ks[k] = kev_scalars(p, m)
        tm["kev"] = time.time() - t
        pa = None
        if ar is not None:
            t = time.time()
            pa = [d["p_yes"] for d in ar.score([ar_messages(batch[i]["problem"], clip_solution(sb.tok, c["text"])) for i, _, c in items], "noul")]
            tm["ar"] = time.time() - t
        P, G = len(batch), a.group
        use = np.ones(len(items), dtype=bool)
        sub_log = {}
        if a.cond in SUB:
            lab = np.zeros(len(items), dtype=bool)
            rs = np.random.default_rng(a.seed * 7919 + step)
            for i in range(P): lab[i * G + rs.choice(G, max(1, round(a.label_frac * G)), replace=False)] = True
            R = v.astype(float).copy()   # labelled: the verifier; unlabelled: the fill-in reward below
            unl = np.flatnonzero(~lab)
            if a.cond == "rlvr_sub":
                use = lab
            elif a.cond == "sub_kev":
                R[unl] = [ks[k]["correct"] for k in unl]
            else:
                t = time.time()
                recal.add([feats[k] for k in np.flatnonzero(lab)], v[lab])
                sub_log["recal_loss"] = recal.fit()
                R[unl] = recal.p_yes([feats[k] for k in unl])
                tm["recal"] = time.time() - t
            if a.cond != "rlvr_sub":
                fill, truth = R[unl], v[unl]
                sub_log.update({"fill_auroc_unlabelled": auroc(fill, truth), "fill_ece_unlabelled": ece(fill, truth),
                                "fill_mean_unlabelled": float(fill.mean()), "acc_unlabelled": float(truth.mean()),
                                "frozen_kev_auroc_unlabelled": auroc([ks[k]["correct"] for k in unl], truth)})
        else:
            R = np.array([reward(a.cond, v[k], ks[k], pa[k] if pa else None, a.lam) for k in range(len(items))])
        Rm = np.where(use, R, np.nan).reshape(P, G)
        with np.errstate(invalid="ignore"):
            Cn = np.nan_to_num((Rm - np.nanmean(Rm, 1, keepdims=True)).reshape(-1))
        A = Cn / (Cn[use].std() + 1e-4)

        # policy update (one step on the whole batch)
        t = time.time()
        n_tok = sum(len(c["token_ids"]) for k, (_, _, c) in enumerate(items) if use[k])   # tokens of the completions in the update
        upd = policy_update(sb, opt, params, [(p, c["token_ids"], c["logprobs"]) for _, p, c in items], A, n_tok, a)
        gn = upd["grad_norm"]
        tm["train"] = time.time() - t

        Vg = v.reshape(P, G)
        grp = np.repeat(np.arange(P), G)
        pc, vs, gs = np.array([ks[k]["correct"] for k in scored]), v[scored], grp[scored]
        wg, n_mixed = within_group_auroc(pc, vs, gs)
        Ls = np.array([len(c["token_ids"]) for _, _, c in items])
        rec = {"kind": "step", "step": step, "epoch": sampler.epoch, "reward_mean": float(R.mean()), "acc": float(v.mean()),
               "kev": {q: float(np.mean([ks[k][q] for k in scored])) for q in QUESTIONS}, "kev_scored": len(scored),
               "kev_correct_auroc": auroc(pc, vs), "kev_correct_auroc_within_group": wg, "groups_mixed": n_mixed,
               "groups_all_wrong": float((Vg.max(1) == 0).mean()), "groups_all_right": float((Vg.min(1) == 1).mean()),
               "groups_zero_reward_var": float((np.nan_to_num(np.nanstd(np.where(use, R, np.nan).reshape(P, G), 1)) == 0).mean()),
               "adv_scale": float(Cn[use].std()), **sub_log,
               "len_mean": float(Ls.mean()), "len_max": int(Ls.max()),
               "truncated": float(np.mean([c["finish_reason"] == "length" for _, _, c in items])),
               "tokens": int(n_tok), **upd,
               "mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "seconds": {k: round(x, 2) for k, x in tm.items()}}
        rec["gen_tokens"] = int(Ls.sum())
        if tree is not None:
            rec.update(tree_log(tree, items, v, verifier, batch))
        if pa is not None:
            rec["ar_correct_mean"], rec["ar_correct_auroc"] = float(np.mean(pa)), auroc(pa, v)
            rec["ar_correct_auroc_within_group"] = within_group_auroc(pa, v, grp)[0]
        emit(rec)
        if roll is not None:
            for k, (i, _, c) in enumerate(items):
                roll.write(json.dumps({"step": step, "id": batch[i]["id"], "k": k % G, "text": c["text"], "n_tok": len(c["token_ids"]),
                                       "finish": c["finish_reason"], "v": int(v[k]), "kev": ks[k], "ar": pa[k] if pa else None,
                                       "r": float(R[k]), "adv": float(A[k]), "used": bool(use[k]),
                                       "labelled": bool(lab[k]) if a.cond in SUB else True, "root": c.get("root")}) + "\n")
            roll.flush()
        print(f"step {step} acc {v.mean():.3f} R {R.mean():.3f} Pc {pc.mean():.3f} auroc {rec['kev_correct_auroc']} "
              f"len {Ls.mean():.0f} gn {gn:.3f} t {rec['seconds']}", flush=True)
        if (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            save_ckpt(out, sb, opt, sampler, step + 1, recal)

    final = {"config": vars(a)}
    if a.final_eval:
        sets = {"gsm8k_test": (gsm, 1, 0.0, 1.0), "math500": math500,
                "amc23": (load(f"{a.data}/amc23.jsonl"), 16, 0.6, 0.95), "aime24": (load(f"{a.data}/aime24.jsonl"), 16, 0.6, 0.95)}
        if Path(f"{a.data}/aime25.jsonl").exists():
            sets["aime25"] = (load(f"{a.data}/aime25.jsonl"), 16, 0.6, 0.95)
        with gzip.open(out / "final_completions.jsonl.gz", "wt") as dump:
            final["eval"] = evaluate(ro, lora, sets, verifier, a.eval_max_tokens, sb=sb, judge_sets=("math500",), dump=dump)
        emit({"kind": "final", "step": a.steps, **final["eval"]})
    sb.save_policy(out / "policy_final")
    (out / "final.json").write_text(json.dumps(final, indent=1))
    if roll is not None: roll.close()
    for d in ("lora_hf", "lora_vllm_0", "lora_vllm_1", "ckpt"):
        if (out / d).exists(): shutil.rmtree(out / d)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
