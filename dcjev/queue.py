"""Run a list of GPU jobs with bounded concurrency and a free-memory gate, skipping the finished ones.

    python -m dcjev.queue --jobs scripts/jobs/part2_main.txt --workers 3 --name part2_main

Each non-comment line of the jobs file is `<out_dir> <free_gb> <command ...>`. A job is finished when <out_dir>/final.json
exists, and is then skipped, so the queue can be started again after an interruption (GRPO jobs resume from their
checkpoints). The first job in file order whose <free_gb> GB are free on the GPU is launched when fewer than --workers jobs of this queue
are running and the previous launch was at least --settle seconds ago (so that its memory has grown before the next
check); a job that needs more memory does not hold up the smaller ones behind it.
A job that exits without final.json is logged to <log_dir>/<name>.failures.jsonl and not retried.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return float(out.stdout.split()[0]) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--name", required=True)
    ap.add_argument("--log_dir", default="/content/work/logs")
    ap.add_argument("--settle", type=float, default=240.0)
    a = ap.parse_args()
    jobs = []
    for line in Path(a.jobs).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        out, gb, cmd = line.split(None, 2)
        jobs.append((Path(out), float(gb), cmd))
    todo = [j for j in jobs if not (j[0] / "final.json").exists()]
    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    logs = Path(a.log_dir); logs.mkdir(parents=True, exist_ok=True)
    running, last = [], 0.0
    while todo or running:
        fit = None
        if todo and len(running) < a.workers and time.time() - last >= a.settle:
            free = free_gb()
            fit = next((k for k, j in enumerate(todo) if free >= j[1]), None)
        if fit is not None:
            out, gb, cmd = todo.pop(fit)
            out.mkdir(parents=True, exist_ok=True)
            tag = "_".join(out.parts[-2:])
            f = open(logs / f"{a.name}.{tag}.log", "a")
            p = subprocess.Popen(["bash", "-c", f"source /content/work/env.sh; cd /content/work/code/DC-Jev-RL; {cmd}"],
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            running.append((p, out, f, cmd)); last = time.time()
            print(f"{time.strftime('%H:%M:%S')} start {out} (free {free_gb():.1f} GB before)", flush=True)
        still = []
        for p, out, f, cmd in running:
            if p.poll() is None:
                still.append((p, out, f, cmd)); continue
            f.close()
            ok = (out / "final.json").exists()
            print(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} {out} (exit {p.returncode})", flush=True)
            if not ok:
                with open(logs / f"{a.name}.failures.jsonl", "a") as g:
                    g.write(json.dumps({"out": str(out), "cmd": cmd, "code": p.returncode, "time": time.time()}) + "\n")
        running = still
        time.sleep(10)
    print("QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
