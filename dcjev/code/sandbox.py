"""Execution of model-written Python against unit tests, for Part III-A.

Each program runs in its own `python -I` subprocess in an empty temporary directory, with CPU-time, address-space and
file-size limits and a wall-clock timeout, after the destructive parts of os / shutil / subprocess are disabled (the
"reliability guard" of OpenAI's human-eval execution.py, MIT licence). This is containment for buggy code from a small
model, not a security boundary against an adversary.

run_tests(code, tests) runs each test (an assert statement or a block) separately after the code, so a failing test
does not hide the others, and reports per test: passed / failed (with the exception, as the feedback a turn shows) /
timeout.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

GUARD = r'''
import builtins, faulthandler, os, platform, resource, shutil, subprocess, sys
resource.setrlimit(resource.RLIMIT_AS, (__MEM__, __MEM__))
resource.setrlimit(resource.RLIMIT_DATA, (__MEM__, __MEM__))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
resource.setrlimit(resource.RLIMIT_CPU, (__CPU__, __CPU__))
faulthandler.disable()
os.environ["OMP_NUM_THREADS"] = "1"
try:
    import numpy   # imported before the guard: the EvalPlus harnesses use it and its import calls functions disabled below
except ImportError:
    pass
for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty", "killpg",
             "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot", "lchown",
             "getcwd", "chdir"):
    setattr(os, name, None)
shutil.rmtree = shutil.move = shutil.chown = None
subprocess.Popen = None
builtins.exit = builtins.quit = None
sys.modules["ipdb"] = sys.modules["joblib"] = sys.modules["resource"] = sys.modules["psutil"] = sys.modules["tkinter"] = None
'''

RUNNER = r'''
import json, sys, traceback, signal
src = json.loads(sys.stdin.read())
g = {"__name__": "__main__"}
out = []
try:
    exec(compile(src["code"], "solution.py", "exec"), g)
except BaseException as e:
    print(json.dumps({"load_error": f"{type(e).__name__}: {e}"[:500]})); sys.exit(0)
def _alarm(*_): raise TimeoutError("test timed out")
signal.signal(signal.SIGALRM, _alarm)
for t in src["tests"]:
    signal.alarm(src["per_test_s"])
    try:
        exec(compile(t, "test.py", "exec"), g); out.append(["passed", ""])
    except AssertionError as e:
        out.append(["failed", ("AssertionError: " + str(e))[:300]])
    except TimeoutError:
        out.append(["timeout", "timed out"])
    except BaseException as e:
        out.append(["failed", f"{type(e).__name__}: {e}"[:300]])
    finally:
        signal.alarm(0)
print(json.dumps({"results": out}))
'''


def run_tests(code, tests, per_test_s=3, total_s=20, mem_mb=1024):
    """-> {"load_error": str} if the code itself fails to run, else {"results": [[status, message], ...]}; plus "timeout"
    if the whole program exceeded total_s."""
    prog = GUARD.replace("__MEM__", str(mem_mb << 20)).replace("__CPU__", str(total_s)) + RUNNER
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", prog], input=json.dumps({"code": code, "tests": tests, "per_test_s": per_test_s}),
                               capture_output=True, text=True, timeout=total_s + 5, cwd=d, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return {"timeout": True, "results": [["timeout", "timed out"]] * len(tests)}
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not lines:
        return {"load_error": (p.stderr.strip().splitlines() or ["crashed"])[-1][:300]}
    return json.loads(lines[-1])


def pass_count(res, n):
    if "results" not in res: return 0
    return sum(s == "passed" for s, _ in res["results"][:n])


class Executor:
    """Runs many (code, tests) jobs in parallel subprocesses."""
    def __init__(self, workers=32):
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def __call__(self, jobs, **kw):
        return list(self.pool.map(lambda j: run_tests(j[0], j[1], **kw), jobs))
