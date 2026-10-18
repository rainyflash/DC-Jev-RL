"""vLLM rollout server for the GRPO trainers (runs in the vLLM environment; the trainers run in kev's environment).

    python -m dcjev.llm.rollout_server --port 6200 --gpu_memory_utilization 0.30

Request (multiprocessing.connection, pickled dict):
  {"op": "generate", "prompts": [str] (or "prompt_ids": [[int]], token ids used as given), "n": int, "temperature": float, "top_p": float, "max_tokens": int, "stop": [str],
   "seeds": None | [int] (one per prompt; vLLM gives the n samples of a prompt seeds s, s+1, ...),
   "lora": None | {"name": str, "path": str}, "top_logprobs": 0 | k (also return the k most likely tokens per position)}
Response: {"prompt_token_ids": [[int]], "outputs": [[{"text", "token_ids", "logprobs", "finish_reason", "top"}] * n], "seconds"}
`logprobs` are vLLM's log-probabilities of the sampled tokens (raw model distribution; the trainers sample at T=1,
top_p=1, where it is also the sampling distribution), used for the truncated importance-sampling correction.
A LoRA is identified by its name: every new name gets a new vLLM adapter id, so a trainer publishes each update under a
new name. The LoRA path must hold an adapter already renamed by dcjev.llm.lora_io.to_vllm.
"""
from __future__ import annotations

import argparse
import time
import traceback
from multiprocessing.connection import Listener
from threading import Lock, Thread

AUTHKEY = b"dcjev-rollout"
BASE, REV = "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--revision", default=REV)
    ap.add_argument("--no_lora", action="store_true", help="serve the model as is (e.g. the generative-judge baseline)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.30)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--max_loras", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_num_seqs", type=int, default=512)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    lora = {} if a.no_lora else dict(enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=a.max_loras, max_cpu_loras=2 * a.max_loras)
    llm = LLM(model=a.model, revision=a.revision, gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
              max_num_seqs=a.max_num_seqs, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, **lora)
    ids, lock = {}, Lock()   # LoRA name -> vLLM id; one generate call at a time (several trainers may be connected)

    def serve(conn):
        try:
            while True:
                m = conn.recv()
                if m.get("op") == "ping":
                    conn.send({"ok": True}); continue
                try:
                    with lock:
                        t0 = time.time()
                        base = dict(n=m["n"], temperature=m["temperature"], top_p=m.get("top_p", 1.0), max_tokens=m["max_tokens"],
                                    stop=m.get("stop"), logprobs=m.get("top_logprobs", 0) if m.get("logprobs", True) else None)
                        seeds = m.get("seeds")
                        sp = [SamplingParams(**base, seed=s) for s in seeds] if seeds else SamplingParams(**base, seed=m.get("seed"))
                        lr = None
                        if m.get("lora"):
                            name = m["lora"]["name"]
                            lr = LoRARequest(name, ids.setdefault(name, len(ids) + 1), m["lora"]["path"])
                        prompts = [{"prompt_token_ids": p} for p in m["prompt_ids"]] if m.get("prompt_ids") is not None else m["prompts"]
                        res = llm.generate(prompts, sp, lora_request=lr, use_tqdm=False)
                        outs = [[{"text": c.text, "token_ids": list(c.token_ids), "finish_reason": c.finish_reason,
                                  "logprobs": [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)] if c.logprobs else None,
                                  "top": [{k: v.logprob for k, v in d.items()} for d in c.logprobs] if m.get("top_logprobs") else None}
                                 for c in r.outputs] for r in res]
                        reply = {"prompt_token_ids": [list(r.prompt_token_ids) for r in res], "outputs": outs, "seconds": time.time() - t0}
                    conn.send(reply)
                except Exception:
                    conn.send({"error": traceback.format_exc()})
        except (EOFError, ConnectionResetError, BrokenPipeError):
            conn.close()

    print(f"rollout server ready on port {a.port}: {a.model}@{a.revision}", flush=True)
    with Listener(("127.0.0.1", a.port), authkey=AUTHKEY) as lst:
        while True:
            Thread(target=serve, args=(lst.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
