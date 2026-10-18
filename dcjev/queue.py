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
A job whose trainer is still running when the queue starts (left by an earlier queue process, e.g. one replaced by a
queue with a new job list) is adopted, not started again: it counts towards --workers and --budget_gb, and its end is
checked like that of a job this queue started.
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


def is_trainer(pid, out):
    """Whether process <pid> is a running dcjev trainer (python -m dcjev....) whose arguments name <out>."""
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X") and str(out).encode() in argv and any(x.startswith(b"dcjev.") for x in argv)


def job_pid(out):
    """The process id of a running dcjev trainer whose arguments name <out> (None if there is none). A trainer's worker
    processes (forked, so with the same arguments) also match; the one returned is the trainer itself, whose parent
    does not match."""
    pids = {int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit() and is_trainer(p.name, out)}
    def ppid(pid):
        try:
            return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            return None
    roots = sorted(pid for pid in pids if ppid(pid) not in pids)
    return roots[0] if roots else None


class Adopted:
    """A trainer started by an earlier queue process, followed through /proc (it is not a child of this process)."""
    def __init__(self, pid, out):
        self.pid, self.out, self.returncode = pid, out, None

    def poll(self):
        if self.returncode is None and not is_trainer(self.pid, self.out):
            self.returncode = "unknown (adopted)"
        return self.returncode


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
    running, last = [], 0.0
    for j in list(todo):
        pid = job_pid(j[0])
        if pid is not None:
            todo.remove(j); running.append((Adopted(pid, j[0]), j[0], None, j[2], j[1]))
            print(f"{time.strftime('%H:%M:%S')} adopted {j[0]} (pid {pid}, started by an earlier queue)", flush=True)
    print(f"{len(todo)} of {len(jobs)} jobs to run, {len(running)} running already", flush=True)
    logs = Path(a.log_dir); logs.mkdir(parents=True, exist_ok=True)
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
            if f is not None: f.close()
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
