"""Run a list of GPU jobs with bounded concurrency and a memory budget, skipping the finished ones.

    python -m dcjev.queue --jobs scripts/jobs/part2_main.txt --workers 3 --budget_gb 73 --name part2_main

Each non-comment line of the jobs file is `<out_dir> <gb> [h=<hours>] <command ...>`, where <gb> is the job's peak GPU
memory (GiB, as nvidia-smi counts it) and <hours> its expected duration. A job is finished when <out_dir>/final.json
exists, and is then skipped, so the queue can be started again after an interruption (GRPO jobs resume from their
checkpoints).
The first job in file order that fits is launched when fewer than --workers jobs of this queue are running and the
previous launch was at least --settle seconds ago. A job fits when the peaks of the running jobs plus its own stay within
--budget_gb (the memory left to this queue by the processes it does not manage, e.g. the vLLM server), and when its peak
is free on the GPU right now. The budget matters because a trainer reaches its peak only after its first evaluation,
minutes after it starts. A job that needs more memory does not hold up the smaller ones behind it.
A job that exits without final.json is logged to <log_dir>/<name>.failures.jsonl and not retried.
With --deadline (unix time, e.g. the end of a Colab runtime) no job is started later than its expected duration
(h=, or --job_hours) before it; the jobs that are not started are listed, to be run elsewhere.
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
    ap.add_argument("--budget_gb", type=float, default=0.0, help="GPU memory for this queue's jobs (0: no budget, free memory only)")
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time after which no job may still be running (0: none)")
    ap.add_argument("--job_hours", type=float, default=7.0, help="expected duration of a job without h=, for --deadline")
    a = ap.parse_args()
    jobs = []
    for line in Path(a.jobs).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        out, gb, cmd = line.split(None, 2)
        hours = a.job_hours
        if cmd.startswith("h="):
            h, cmd = cmd.split(None, 1); hours = float(h[2:])
        jobs.append((Path(out), float(gb), cmd, hours))
    todo = [j for j in jobs if not (j[0] / "final.json").exists()]
    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    logs = Path(a.log_dir); logs.mkdir(parents=True, exist_ok=True)
    running, last = [], 0.0
    while todo or running:
        late = [j for j in todo if a.deadline and time.time() + 3600 * j[3] > a.deadline]
        if late:
            print(f"{time.strftime('%H:%M:%S')} deadline: not starting " + " ".join(str(j[0]) for j in late), flush=True)
            todo = [j for j in todo if j not in late]
        fit = None
        if todo and len(running) < a.workers and time.time() - last >= a.settle:
            free, used = free_gb(), sum(r[4] for r in running)
            fit = next((k for k, j in enumerate(todo) if free >= j[1] and (not a.budget_gb or used + j[1] <= a.budget_gb)), None)
        if fit is not None:
            out, gb, cmd, _ = todo.pop(fit)
            out.mkdir(parents=True, exist_ok=True)
            tag = "_".join(out.parts[-2:])
            f = open(logs / f"{a.name}.{tag}.log", "a")
            p = subprocess.Popen(["bash", "-c", f"source /content/work/env.sh; cd /content/work/code/DC-Jev-RL; {cmd}"],
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            running.append((p, out, f, cmd, gb)); last = time.time()
            print(f"{time.strftime('%H:%M:%S')} start {out} (free {free:.1f} GiB, budget used {used + gb:.0f} GiB)", flush=True)
        still = []
        for r in running:
            p, out, f, cmd, _ = r
            if p.poll() is None:
                still.append(r); continue
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
