"""Checks the rollout server's handling of concurrent requests against a reference taken one request at a time.

    python -m dcjev.llm.check_rollout_server ref   --port 6200 --lora /content/work/tmp/lora_rand_vllm --out /content/work/tmp/rollout_ref.json
    python -m dcjev.llm.check_rollout_server check --port 6200 --lora /content/work/tmp/lora_rand_vllm --out /content/work/tmp/rollout_ref.json

ref    four fixed requests (MATH-500 prompts: seeded samples at T = 1 without and with a LoRA, greedy decoding, and
       continuations of given token prefixes), sent one at a time; outputs saved.
check  the same requests sent one at a time, then all at once from separate connections. Per request: whether every
       reply carries the right prompts, the fraction of samples identical to the reference, and the mean absolute
       difference of the log-probabilities on the tokens before the first divergence. Also the wall-clock of both modes.
Samples need not be bit-identical across batch compositions (bf16 kernels), so divergence is reported, not asserted;
a reply routed to the wrong request or the wrong adapter would show as near-zero agreement.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dcjev.llm.data import PROMPT, STOP, load


def requests(data, lora, prefixes=None):
    m5 = load(f"{data}/math500.jsonl")
    P = [PROMPT.format(problem=r["problem"]) for r in m5]
    out = {"sampled": dict(prompts=P[:16], n=4, temperature=1.0, top_p=1.0, max_tokens=256, stop=STOP, seeds=[11 * i for i in range(16)], lora=None),
           "sampled_lora": dict(prompts=P[16:32], n=4, temperature=1.0, top_p=1.0, max_tokens=256, stop=STOP, seeds=[13 * i for i in range(16)],
                                lora={"name": "check-rand", "path": lora} if lora else None),
           "greedy": dict(prompts=P[32:64], n=1, temperature=0.0, top_p=1.0, max_tokens=128, stop=STOP, lora=None)}
    if prefixes:
        out["continued"] = dict(prompt_ids=prefixes, n=2, temperature=1.0, top_p=1.0, max_tokens=128, stop=STOP, seeds=[17 * i for i in range(len(prefixes))], lora=None)
    return out


def send(port, req):
    from dcjev.llm.grpo import Rollouts
    t0 = time.time()
    r = Rollouts(port)(**req)
    return r, time.time() - t0


def compare(ref, got):
    same, diffs, right = [], [], ref["prompt_token_ids"] == got["prompt_token_ids"]
    for ro, go in zip(ref["outputs"], got["outputs"]):
        for a, b in zip(ro, go):
            same.append(a["token_ids"] == b["token_ids"])
            k = 0
            while k < min(len(a["token_ids"]), len(b["token_ids"])) and a["token_ids"][k] == b["token_ids"][k]: k += 1
            if a["logprobs"] and b["logprobs"] and k:
                diffs.append(float(np.mean(np.abs(np.array(a["logprobs"][:k]) - np.array(b["logprobs"][:k])))))
    return {"right_prompts": right, "identical_frac": float(np.mean(same)), "samples": len(same),
            "lp_absdiff_before_divergence": float(np.mean(diffs)) if diffs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "check"])
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--lora", default="")
    ap.add_argument("--data", default="/content/work/data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "ref":
        reqs = requests(a.data, a.lora)
        ref = {k: send(a.port, r)[0] for k, r in reqs.items()}
        prefixes = [p + o[0]["token_ids"][:64] for p, o in zip(ref["sampled"]["prompt_token_ids"], ref["sampled"]["outputs"])]
        ref["continued"] = send(a.port, requests(a.data, a.lora, prefixes)["continued"])[0]
        json.dump({"prefixes": prefixes, "ref": ref}, open(a.out, "w"))
        print("reference saved", {k: len(v["outputs"]) for k, v in ref.items()}, flush=True)
        return
    saved = json.load(open(a.out))
    reqs = requests(a.data, a.lora, saved["prefixes"])
    rep = {"sequential": {}, "concurrent": {}}
    t0 = time.time()
    for k, r in reqs.items():
        got, s = send(a.port, r)
        rep["sequential"][k] = {**compare(saved["ref"][k], got), "seconds": round(s, 2)}
    rep["sequential_seconds"] = round(time.time() - t0, 2)
    t0 = time.time()
    with ThreadPoolExecutor(len(reqs)) as ex:
        res = dict(zip(reqs, ex.map(lambda r: send(a.port, r), reqs.values())))
    rep["concurrent_seconds"] = round(time.time() - t0, 2)
    for k, (got, s) in res.items():
        rep["concurrent"][k] = {**compare(saved["ref"][k], got), "seconds": round(s, 2)}
    json.dump(rep, open(a.out.replace(".json", "_check.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
