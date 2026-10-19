"""Part II systems check: Kev-4B on the shared backbone (policy LoRA + Kev's LoRA on one Qwen3.5-4B-Base) against Kev-4B
loaded on its own, and the memory and time each arrangement costs.

    python -m dcjev.llm.check_shared --out /content/work/runs/part2/shared_check.json

parity  Kev's probabilities on the same requests from (a) the shared backbone (unmerged fp32 adapter on the bf16 base,
        shared-prefix scoring, what the GRPO trainer uses), (b) Kev on its own, bf16 merged + fused kernels (the Part I
        judge), against (c) kev's fp32 reference path (DecisionModel.probs). Requests: math solutions in the Part II judge
        format (GSM8K and MATH reference solutions, and the same solutions with the final answer changed).
memory  CUDA memory, each arrangement in a fresh process: (a) one backbone carrying both adapters + Kev's head; (b) the
        policy model (backbone + policy LoRA) and Kev on its own (second backbone, merged). After loading, and the peak
        over one policy training micro-batch plus one Kev scoring batch.
speed   Kev scoring throughput in each arrangement; the cost of switching the active adapter.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch

from dcjev.metrics import auroc


def gsm_solution(sol):
    body, ans = sol.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return body, ans.strip().replace(",", "")


def make_requests(n_each=48, seed=0):
    """(request, label): label 1 = the reference solution as written, 0 = the same solution with its final answer changed."""
    from datasets import load_dataset
    from dcjev.llm.data import MATH_SUBJECTS, last_boxed
    from dcjev.llm.prompts import kev_request
    rng = random.Random(seed)
    out = []
    g = load_dataset("openai/gsm8k", "main", split="train")
    for i in rng.sample(range(len(g)), n_each):
        body, ans = gsm_solution(g[i]["answer"])
        wrong = str(int(float(ans)) + rng.choice([-3, -2, -1, 1, 2, 3, 10])) if re.fullmatch(r"-?\d+(\.\d+)?", ans) else ans + "1"
        for a, lab in ((ans, 1), (wrong, 0)):
            out.append((kev_request(g[i]["question"], f"{body}\nThe final answer is $\\boxed{{{a}}}$."), lab))
    m = load_dataset("EleutherAI/hendrycks_math", rng.choice(MATH_SUBJECTS), split="train")
    for i in rng.sample(range(len(m)), n_each):
        sol = m[i]["solution"]; ans = last_boxed(sol)
        if ans is None: continue
        wrong = str(int(ans) + rng.choice([-2, -1, 1, 2])) if re.fullmatch(r"-?\d+", ans) else ans + "+1"
        k = sol.rfind(ans)
        for a, lab in ((ans, 1), (wrong, 0)):
            out.append((kev_request(m[i]["problem"], sol[:k] + a + sol[k + len(ans):]), lab))
    return out


def part_parity(a):
    from dcjev.judges.kev_judge import KevJudge
    from dcjev.llm.shared import KEV_RUN, SharedBackbone
    data = make_requests()
    reqs, labels = [r for r, _ in data], [l for _, l in data]
    res = {"requests": len(reqs)}
    ref = KevJudge(KEV_RUN, dtype="fp32", merge=False)
    p_ref = [ref.reference_probs(r) for r in reqs]
    del ref; torch.cuda.empty_cache()
    def compare(p):
        d, agree, total = [], 0, 0
        for x, y in zip(p, p_ref):
            for q1, q2 in zip(x, y):
                d.append(float(np.abs(np.array(q1) - np.array(q2)).max())); agree += int(np.argmax(q1) == np.argmax(q2)); total += 1
        return {"questions": total, "max_abs_diff": max(d), "mean_max_abs_diff": float(np.mean(d)),
                "p99_max_abs_diff": float(np.quantile(d, 0.99)), "argmax_agreement": agree / total}
    sep = KevJudge(KEV_RUN, dtype="bf16", merge=True, fused=True)
    p_sep, _ = sep.probs(reqs)
    res["separate_bf16_merged_fused"] = compare(p_sep)
    del sep; torch.cuda.empty_cache()
    sb = SharedBackbone()
    p_sh, _ = sb.kev_probs(reqs)
    res["shared_backbone_unmerged"] = compare(p_sh)
    ci = 0   # the 'correct' question comes first in the request
    res["sanity_auroc_correct_vs_altered_answer"] = {
        "fp32_reference": auroc([p[ci][1] for p in p_ref], labels), "shared_backbone": auroc([p[ci][1] for p in p_sh], labels),
        "separate_bf16": auroc([p[ci][1] for p in p_sep], labels)}
    return res


def batch_tokens(tok, n_seq=8, L=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(1000, 100000, (128,), generator=g).tolist(), torch.randint(1000, 100000, (L - 128,), generator=g).tolist()) for _ in range(n_seq)]


def part_memory(a):
    """One arrangement in this process: CUDA memory after load, peak over a training micro-batch and a Kev batch, Kev speed."""
    from dcjev.llm.check_shared import make_requests as mk
    reqs = [r for r, _ in mk(n_each=64, seed=1)]
    torch.cuda.reset_peak_memory_stats()
    res = {"arrangement": a.part}
    if a.part == "mem_shared":
        from dcjev.llm.shared import SharedBackbone
        sb = SharedBackbone()
        res["after_load_gb"] = torch.cuda.memory_allocated() / 1e9
        def train_step():
            sb.model.train()
            lps = sb.token_logprobs(batch_tokens(sb.tok))
            sum(lp.sum() for lp in lps).backward()
            for p in sb.policy_parameters(): p.grad = None
        score = lambda rs: sb.kev_probs(rs)
        t = time.perf_counter()
        for _ in range(100): sb.use("kev"); sb.use("default")
        res["adapter_switch_round_trip_ms"] = (time.perf_counter() - t) / 100 * 1e3
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dcjev.judges.kev_judge import KevJudge
        from dcjev.llm.shared import BASE, BASE_REV, KEV_RUN, POLICY_TARGETS
        tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
        lm = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REV, dtype=torch.bfloat16).cuda()
        pm = get_peft_model(lm, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=POLICY_TARGETS, task_type="CAUSAL_LM"))
        pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        kev = KevJudge(KEV_RUN, dtype="bf16", merge=True, fused=True, shared_prefix=False)
        res["after_load_gb"] = torch.cuda.memory_allocated() / 1e9
        def train_step():
            pm.train()
            seqs = batch_tokens(tok)
            ids = torch.tensor([p + c for p, c in seqs], device="cuda")
            h = pm.base_model.model.model(input_ids=ids).last_hidden_state
            W = pm.base_model.model.lm_head.weight
            from dcjev.llm.shared import _lp_chunk
            from torch.utils.checkpoint import checkpoint
            hf, tgt = h[:, 127:-1].reshape(-1, h.shape[-1]), ids[:, 128:].reshape(-1)
            sum(checkpoint(_lp_chunk, hf[s:s + 1024], tgt[s:s + 1024], W, use_reentrant=False).sum() for s in range(0, len(tgt), 1024)).backward()
            for p in pm.parameters(): p.grad = None
        score = lambda rs: kev.probs(rs)
    train_step(); torch.cuda.synchronize()
    res["peak_after_train_step_gb"] = torch.cuda.max_memory_allocated() / 1e9
    score(reqs[:32]); torch.cuda.synchronize()
    t = time.perf_counter(); score(reqs); torch.cuda.synchronize(); dt = time.perf_counter() - t
    res["kev_requests_per_s"] = len(reqs) / dt
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all", choices=["all", "parity", "mem_shared", "mem_separate"])
    a = ap.parse_args()
    if a.part == "all":
        report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
        for part in ("parity", "mem_shared", "mem_separate"):
            tmp = f"{a.out}.{part}.json"
            subprocess.run([sys.executable, "-m", "dcjev.llm.check_shared", "--part", part, "--out", tmp], check=True)
            report[part] = json.load(open(tmp))
        s, p = report["mem_separate"], report["mem_shared"]
        report["memory_saving_after_load"] = 1 - p["after_load_gb"] / s["after_load_gb"]
        report["memory_saving_peak"] = 1 - p["peak_gb"] / s["peak_gb"]
        json.dump(report, open(a.out, "w"), indent=1)
        print(json.dumps(report, indent=1))
        return
    res = part_parity(a) if a.part == "parity" else part_memory(a)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
