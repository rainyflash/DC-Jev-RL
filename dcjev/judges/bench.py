"""What one judgment costs, measured on an otherwise idle GPU: Kev-4B in each way this project runs it, the AR judge, and
the reasoning (generative) judge, all on the same Part II requests.

    python -m dcjev.judges.bench --out /content/work/runs/bench/judge_bench.json [--n 1000]   # vLLM env for 'genjudge'

Requests: the first --n solutions of the Part II judge study (experiments/part2_llm/judge_study/completions.jsonl.gz,
written by Qwen3.5-4B-Base), clipped as in GRPO, with Part II's three questions (correct, genuine, quality) or with
'correct' only.
  kev_rows_fused       bf16 merged + kev's fused kernels, one row per question (Part I's judge server, kev.serve's default)
  kev_shared_prefix    bf16 merged, unfused: each state once, its questions as branches (the Part II judge study)
  kev_backbone_8k/32k  Kev's adapter unmerged on the policy's backbone next to the policy LoRA (the GRPO trainers), with
                       the trainers' 8192-token batches and with 32768
  ar_fused             Qwen3.5-4B, kev's fused kernels, one forward pass per question ('correct' as yes/no, 'quality' as
                       a 4-level score)
  genjudge             Qwen3.5-4B thinking (T=0.6, top-p 0.95, up to 2048 tokens) then answering yes/no, served by
                       vLLM (dcjev.llm.rollout_server --no_lora) with all requests submitted at once
Each arrangement runs in its own process after a warm-up batch; GPU utilisation and memory of other processes are
recorded before and after (nvidia-smi), so that a measurement taken next to a busy process can be recognised.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments" / "part2_llm" / "judge_study" / "completions.jsonl.gz"
KEV_PARTS = ["kev_rows_fused", "kev_shared_prefix", "kev_backbone_8k", "kev_backbone_32k"]
AR_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def gpu_state():
    q = lambda f: subprocess.run(["nvidia-smi", f"--query-{f}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    util, used = q("gpu=utilization.gpu,memory.used").split(", ")
    return {"utilization_pct": int(util), "memory_used_mib": int(used),
            "compute_apps": [l for l in q("compute-apps=pid,used_memory").splitlines() if l]}


def solutions(n):
    from transformers import AutoTokenizer
    from dcjev.llm.prompts import clip_solution
    from dcjev.llm.shared import BASE, BASE_REV
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    with gzip.open(STUDY, "rt", encoding="utf-8") as f:
        comps = [json.loads(l) for _, l in zip(range(n), f)]
    return [(c["problem"], clip_solution(tok, c["text"])) for c in comps]


def timed(fn, items, warm=64):
    import torch
    fn(items[:warm]); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); fn(items); torch.cuda.synchronize()
    return time.perf_counter() - t, torch.cuda.max_memory_allocated() / 1e9


def part_kev(name, sols):
    from dcjev.llm.prompts import QUESTIONS, kev_request
    from dcjev.llm.shared import KEV_RUN
    if name.startswith("kev_backbone"):
        from dcjev.llm.shared import SharedBackbone
        sb = SharedBackbone(kev_token_budget=8192 if name.endswith("8k") else 32768)
        score = sb.kev_probs
    else:
        from dcjev.judges.kev_judge import KevJudge
        kev = KevJudge(KEV_RUN, dtype="bf16", merge=True, fused=name == "kev_rows_fused", shared_prefix=name == "kev_shared_prefix")
        score = kev.probs
    res = {}
    for qs in (QUESTIONS, ("correct",)):
        reqs = [kev_request(p, s, questions=qs) for p, s in sols]
        sec, peak = timed(score, reqs)
        res["+".join(qs)] = {"requests": len(reqs), "questions": len(reqs) * len(qs), "seconds": sec, "requests_per_s": len(reqs) / sec,
                             "seconds_per_1k_questions": sec / (len(reqs) * len(qs)) * 1e3, "peak_gb": peak}
    return res


def part_ar(sols):
    from dcjev.judges.ar_judge import ARJudge
    from dcjev.llm.prompts import ar_messages
    ar = ARJudge("Qwen/Qwen3.5-4B", revision=AR_REV, fused=True, token_budget=8192)
    res = {}
    for q, kind in (("correct", "noul"), ("quality", "score")):
        msgs = [ar_messages(p, s, q) for p, s in sols]
        sec, peak = timed(lambda m: ar.score(m, kind, levels=4), msgs)
        res[q] = {"prompts": len(msgs), "seconds": sec, "seconds_per_1k_questions": sec / len(msgs) * 1e3, "peak_gb": peak}
    return res


def part_genjudge(sols, port, n):
    """Needs the vLLM environment for the server; the client side only needs the tokenizer."""
    import numpy as np
    from transformers import AutoTokenizer
    from dcjev.llm.grpo import Rollouts
    from dcjev.llm.prompts import ar_messages
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision=AR_REV)
    prompts = []
    for p, s in sols[:n]:
        m = ar_messages(p, s, "correct")
        m[0]["content"] = m[0]["content"].replace("Answer with yes or no only.", "Think it through, then answer with yes or no only.")
        prompts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True))
    ro = Rollouts(port)
    ro(prompts[:8], n=1, temperature=0.6, top_p=0.95, max_tokens=64, stop=["</think>"], seeds=list(range(8)), lora=None)
    t = time.perf_counter()
    g = ro(prompts, n=1, temperature=0.6, top_p=0.95, max_tokens=2048, stop=["</think>"], seeds=list(range(len(prompts))), lora=None)
    th = [o[0] for o in g["outputs"]]
    ro([p + x["text"] + "\n</think>\n\n" for p, x in zip(prompts, th)], n=1, temperature=0.0, max_tokens=1, top_logprobs=20, lora=None)
    sec = time.perf_counter() - t
    return {"judgments": len(prompts), "seconds": sec, "seconds_per_1k_questions": sec / len(prompts) * 1e3,
            "think_tokens_mean": float(np.mean([len(x["token_ids"]) for x in th])),
            "think_finished": float(np.mean([x["finish_reason"] == "stop" for x in th]))}


def run_child(part, a, tmp):
    before = gpu_state()
    subprocess.run([sys.executable, "-m", "dcjev.judges.bench", "--part", part, "--n", str(a.n), "--out", str(tmp)], check=True)
    return {**json.load(open(tmp)), "gpu_before": before, "gpu_after": gpu_state()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all", choices=["all", *KEV_PARTS, "ar_fused", "genjudge"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--genjudge_n", type=int, default=300)
    ap.add_argument("--genjudge_port", type=int, default=6300)
    ap.add_argument("--vllm_python", default="/content/work/vllm-env/bin/python")
    a = ap.parse_args()
    out = Path(a.out)
    if a.part != "all":
        sols = solutions(a.n)
        res = part_ar(sols) if a.part == "ar_fused" else part_kev(a.part, sols)
        json.dump(res, open(out, "w"), indent=1); return
    import torch   # no CUDA in this process: the children and the server each measure alone
    name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    rep = {"gpu": name, "torch": torch.__version__, "n": a.n, "requests": str(STUDY.relative_to(ROOT))}
    out.parent.mkdir(parents=True, exist_ok=True)
    for part in [*KEV_PARTS, "ar_fused"]:
        rep[part] = run_child(part, a, out.with_suffix(f".{part}.json"))
        json.dump(rep, open(out, "w"), indent=1); print(part, json.dumps(rep[part])[:400], flush=True)
    env = {**os.environ, "PATH": f"{Path(a.vllm_python).parent}:{os.environ['PATH']}"}
    log = open(out.with_suffix(".genjudge_server.log"), "w")
    srv = subprocess.Popen([a.vllm_python, "-m", "dcjev.llm.rollout_server", "--port", str(a.genjudge_port), "--model", "Qwen/Qwen3.5-4B",
                            "--revision", AR_REV, "--no_lora", "--gpu_memory_utilization", "0.25", "--max_model_len", "8192"],
                           stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        while "rollout server ready" not in out.with_suffix(".genjudge_server.log").read_text(errors="replace"):
            if srv.poll() is not None: raise RuntimeError("genjudge server exited")
            time.sleep(5)
        before = gpu_state()
        rep["genjudge"] = {**part_genjudge(solutions(a.genjudge_n), a.genjudge_port, a.genjudge_n), "gpu_before": before, "gpu_after": gpu_state()}
    finally:
        srv.terminate(); srv.wait(60)
    json.dump(rep, open(out, "w"), indent=1)
    print("genjudge", json.dumps(rep["genjudge"])[:400], flush=True)


if __name__ == "__main__":
    main()
