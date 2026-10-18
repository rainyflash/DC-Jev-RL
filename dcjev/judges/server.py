"""One GPU process that serves judge scores to many CPU training workers.

    python -m dcjev.judges.server --port 6100 --judges kev,ar

Protocol (multiprocessing.connection over localhost, pickled dicts):
  request  {"judge": "kev" | "ar", "env": env_id, "questions": ["progress", "healthy"], "summaries": [window summary, ...]}
  response {"progress": [...], "progress_conf": [...], "progress_sampled": [...] (ar), "healthy": [...], "valid_mass": {...}}
           progress = expected level / (levels - 1) in [0, 1]; healthy = p(yes).
  request  {"op": "stats"} -> per-judge counters (requests, rows, seconds).
Requests that arrive while the GPU is busy are merged into one batch per judge, so throughput does not depend on how many
workers there are.
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import traceback
from multiprocessing.connection import Listener

import numpy as np

from dcjev.control import prompts
from kev.api import score_confidence

AUTHKEY = b"dcjev-judge"


def kev_scores(judge, items):
    reqs, spans = [], []
    for it in items:
        m = it["msg"]
        start = len(reqs)
        reqs += [prompts.kev_request(m["env"], s, tuple(m["questions"])) for s in m["summaries"]]
        spans.append((start, len(reqs)))
    probs, metas = judge.probs(reqs)
    for it, (a, b) in zip(items, spans):
        qs = it["msg"]["questions"]; res = {q: [] for q in qs}
        if "progress" in qs: res["progress_conf"] = []
        for p, meta in zip(probs[a:b], metas[a:b]):
            for pq, mq in zip(p, meta):
                if mq["id"] == "progress":
                    L = len(pq)
                    res["progress"].append(float(sum(i * x for i, x in enumerate(pq)) / (L - 1)))
                    res["progress_conf"].append(float(score_confidence(pq)))
                else:
                    res["healthy"].append(float(pq[1]))
        it["result"] = res


def ar_scores(judge, items):
    for q in ("progress", "healthy"):
        sel = [it for it in items if q in it["msg"]["questions"]]
        if not sel: continue
        msgs, spans = [], []
        for it in sel:
            start = len(msgs)
            msgs += [prompts.ar_messages(it["msg"]["env"], s, q) for s in it["msg"]["summaries"]]
            spans.append((start, len(msgs)))
        out = judge.score(msgs, "score" if q == "progress" else "noul")
        for it, (a, b) in zip(sel, spans):
            res = it.setdefault("result", {})
            res.setdefault("valid_mass", {})[q] = [o["valid_mass"] for o in out[a:b]]
            if q == "progress":
                res["progress"] = [o["expected"] for o in out[a:b]]
                res["progress_sampled"] = [o["sampled"] for o in out[a:b]]
            else:
                res["healthy"] = [o["p_yes"] for o in out[a:b]]


class Server:
    def __init__(self, judges, port, max_windows=4096, wait_ms=15):
        self.judges, self.port, self.max_windows, self.wait = judges, port, max_windows, wait_ms / 1000
        self.q = queue.Queue()
        self.counters = {k: {"requests": 0, "windows": 0, "seconds": 0.0} for k in judges}

    def serve_client(self, conn):
        try:
            while True:
                msg = conn.recv()
                if msg.get("op") == "stats":
                    conn.send({"counters": self.counters, **{k: vars(getattr(j, "stats", None)) if hasattr(j, "stats") else {} for k, j in self.judges.items()}})
                    continue
                item = {"msg": msg, "done": threading.Event()}
                self.q.put(item); item["done"].wait()
                conn.send(item.get("result") if "error" not in item else {"error": item["error"]})
        except (EOFError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()

    def model_loop(self):
        while True:
            items = [self.q.get()]
            deadline = time.time() + self.wait
            while sum(len(it["msg"]["summaries"]) for it in items) < self.max_windows:
                try:
                    items.append(self.q.get(timeout=max(0.0, deadline - time.time())))
                except queue.Empty:
                    break
            for name in {it["msg"]["judge"] for it in items}:
                group = [it for it in items if it["msg"]["judge"] == name]
                t0 = time.perf_counter()
                try:
                    (kev_scores if name == "kev" else ar_scores)(self.judges[name], group)
                except Exception:
                    err = traceback.format_exc()
                    print(err, flush=True)
                    for it in group: it["error"] = err
                c = self.counters[name]
                c["requests"] += len(group); c["windows"] += sum(len(it["msg"]["summaries"]) for it in group)
                c["seconds"] += time.perf_counter() - t0
            for it in items:
                it["done"].set()

    def run(self):
        threading.Thread(target=self.model_loop, daemon=True).start()
        with Listener(("127.0.0.1", self.port), authkey=AUTHKEY, backlog=256) as lst:
            print(f"judge server ready on port {self.port}: {sorted(self.judges)}", flush=True)
            while True:
                conn = lst.accept()
                threading.Thread(target=self.serve_client, args=(conn,), daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6100)
    ap.add_argument("--judges", default="kev,ar")
    ap.add_argument("--kev_run", default=f"jaredpalmer/kev-4b@{os.environ.get('KEV_REV', '6cfce5c2fa4b4bd64026336ab649c5ca78857d52')}")
    ap.add_argument("--ar_model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--ar_revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--max_windows", type=int, default=4096)
    a = ap.parse_args()
    judges = {}
    if "kev" in a.judges:
        from dcjev.judges.kev_judge import KevJudge
        judges["kev"] = KevJudge(a.kev_run, dtype="bf16", fused=True)   # fastest and closest to fp32 (kev_modes.json)
    if "ar" in a.judges:
        from dcjev.judges.ar_judge import ARJudge
        judges["ar"] = ARJudge(a.ar_model, revision=a.ar_revision, fused=True)
    Server(judges, a.port, a.max_windows).run()


if __name__ == "__main__":
    main()
