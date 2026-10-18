"""Run a grid of PPO jobs in parallel, skipping finished ones (a job is finished when its final.json exists).

    python -m dcjev.control.grid --root runs/part1 --envs all --rewards env,env_window --seeds 1-5 --workers 40

Jobs are ordered seed-major so that a partial grid always covers every condition with the first seeds. A failed job is
logged to <root>/failures.jsonl with its exit code and the tail of its output, and is not retried automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dcjev.control.envs import ENVS


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--envs", default="all")
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--port", type=int, default=6100)
    ap.add_argument("--extra", default="", help="extra arguments passed to every ppo job")
    a = ap.parse_args()
    envs = ENVS if a.envs == "all" else a.envs.split(",")
    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    jobs = [(e, r, s) for s in parse_seeds(a.seeds) for r in a.rewards.split(",") for e in envs]
    jobs = [j for j in jobs if not (root / j[0] / j[1] / f"s{j[2]}" / "final.json").exists()]
    print(f"{len(jobs)} jobs to run", flush=True)
    running, t0 = [], time.time()
    while jobs or running:
        while jobs and len(running) < a.workers:
            e, r, s = jobs.pop(0)
            out = root / e / r / f"s{s}"; out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "dcjev.control.ppo", "--env_id", e, "--reward", r, "--seed", str(s), "--out", str(out),
                   "--port", str(a.port)] + a.extra.split()
            f = open(out / "stdout.txt", "w")
            running.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}), out, f, (e, r, s)))
        time.sleep(5)
        still = []
        for p, out, f, job in running:
            if p.poll() is None:
                still.append((p, out, f, job)); continue
            f.close()
            if p.returncode != 0 or not (out / "final.json").exists():
                tail = (out / "stdout.txt").read_text(errors="replace")[-2000:]
                with open(root / "failures.jsonl", "a") as g:
                    g.write(json.dumps({"job": job, "code": p.returncode, "tail": tail}) + "\n")
                print("FAILED", job, p.returncode, flush=True)
            else:
                print(f"done {job} ({len(jobs)} queued, {len(still)} running, {time.time() - t0:.0f}s)", flush=True)
        running = still
    print("GRID_DONE", flush=True)


if __name__ == "__main__":
    main()
