"""Verifies the trainer -> vLLM LoRA hand-off (run in two steps, one per environment).

  vLLM env: python -m dcjev.llm.check_vllm_lora gen   --lora /content/work/tmp/lora_rand --out /content/work/tmp/vllm_check.json
  kev env:  python -m dcjev.llm.check_vllm_lora score --lora /content/work/tmp/lora_rand --out /content/work/tmp/vllm_check.json

gen: greedy outputs with no adapter, a zero adapter and a random adapter (renamed by dcjev.llm.lora_io.to_vllm); sampled
sequences with their vLLM log-probabilities. score: the same sampled tokens scored by HF/PEFT with the original adapter;
reports the per-token log-probability differences (the train/inference mismatch of the pipeline).
"""
from __future__ import annotations

import argparse
import json

BASE, REV = "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"
PROMPTS = ["Solve the following math problem step by step. Put your final answer within \\boxed{}.\n\nProblem: What is 17 * 23?\n\nSolution:",
           "Solve the following math problem step by step. Put your final answer within \\boxed{}.\n\nProblem: A train travels 60 miles per hour for 3 hours. How far does it go?\n\nSolution:"]


def gen(a):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from dcjev.llm.lora_io import to_vllm
    zero = to_vllm(a.lora.replace("lora_rand", "lora_zero"), a.lora.replace("lora_rand", "lora_zero") + "_vllm")
    rand = to_vllm(a.lora, a.lora + "_vllm")
    llm = LLM(model=BASE, revision=REV, enable_lora=True, max_lora_rank=64, max_loras=2, gpu_memory_utilization=0.25, max_model_len=4096,
              max_num_seqs=256, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    g = SamplingParams(temperature=0.0, max_tokens=48)
    b = llm.generate(PROMPTS, g)
    z = llm.generate(PROMPTS, g, lora_request=LoRARequest("zero", 1, str(zero)))
    r = llm.generate(PROMPTS, g, lora_request=LoRARequest("rand", 2, str(rand)))
    s = llm.generate(PROMPTS, SamplingParams(temperature=1.0, max_tokens=96, n=4, seed=3, logprobs=0), lora_request=LoRARequest("rand", 2, str(rand)))
    samples = [{"prompt_ids": list(o.prompt_token_ids), "ids": list(c.token_ids), "lp": [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]}
               for o in s for c in o.outputs]
    rep = {"zero_equals_base": all(x.outputs[0].token_ids == y.outputs[0].token_ids for x, y in zip(b, z)),
           "rand_differs_from_base": any(x.outputs[0].token_ids != y.outputs[0].token_ids for x, y in zip(b, r)),
           "base_text": b[0].outputs[0].text, "rand_text": r[0].outputs[0].text, "samples": samples}
    json.dump(rep, open(a.out, "w"))
    print({k: v for k, v in rep.items() if k != "samples"})


def score(a):
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    rep = json.load(open(a.out))
    lm = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16).cuda()
    pm = PeftModel.from_pretrained(lm, a.lora).cuda().eval()
    diffs = []
    with torch.no_grad():
        for s in rep["samples"]:
            ids = torch.tensor([s["prompt_ids"] + s["ids"]], device="cuda")
            logits = pm(input_ids=ids).logits[0, len(s["prompt_ids"]) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(s["ids"], device="cuda")[:, None])[:, 0].cpu().numpy()
            diffs += list(np.abs(lp - np.array(s["lp"])))
    d = np.array(diffs)
    rep["hf_vs_vllm_logprob_abs_diff"] = {"mean": float(d.mean()), "p99": float(np.quantile(d, 0.99)), "max": float(d.max()), "tokens": len(d)}
    json.dump(rep, open(a.out, "w"))
    print({k: v for k, v in rep.items() if k != "samples"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "score"])
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (gen if a.step == "gen" else score)(a)
