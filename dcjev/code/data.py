"""Code tasks for Part III-A, normalised to JSONL records
    {id, source, problem, fn, signature, visible: [test code], shown: str, heldout: [test code]}

    python -m dcjev.code.data --out /content/work/data/code

`visible` tests are shown to the policy (in `shown`) and run after every attempt, their failures fed back; `heldout` tests
are never shown and decide whether a solution is correct beyond the tests it was shown. Every test is a self-contained
block run after the solution's code (dcjev.code.sandbox.run_tests).

kodcode_train / kodcode_val  KodCode/KodCode-Light-RL-10K (cc-by-nc-4.0) problems whose pytest file tests one function
            imported from `solution`, uses no fixtures or third-party packages, and has at least 3 test functions; the
            first test function is visible, the others are held out. 300 problems (seed 0) are held out for validation.
mbppplus    evalplus/mbppplus (apache-2.0): the original MBPP asserts (`test_list`: 3 for most problems, up to 7) are visible;
            the EvalPlus harness is held out.
humanevalplus  evalplus/humanevalplus (apache-2.0): the first 3 asserts of the original HumanEval test (openai/openai_humaneval,
            MIT; fewer where it has fewer) are visible; the EvalPlus `check` harness is held out. The problem's own code (signature,
            docstring, helper functions) runs before the solution (`exec_prefix`), so the helpers the harness calls exist;
            the solution's definition replaces the stub.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

KOD_REV = "dcf78a8bbba9a613b596ce993c4921a38687dfcc"
MBPP_REV = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
HE_REV = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
HE_ORIG_REV = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
SAFE_IMPORTS = {"math", "collections", "itertools", "heapq", "functools", "re", "string", "bisect", "random", "typing", "operator",
                "copy", "statistics", "fractions", "decimal", "datetime", "json", "unittest"}


def kodcode_record(r):
    """-> normalised record, or None if the problem's tests do not fit the protocol."""
    info = r["test_info"]
    if len(info) != 1: return None
    fn, decl = info[0]["function_name"], info[0]["function_declaration"]
    try:
        mod = ast.parse(r["test"])
    except SyntaxError:
        return None
    pre, tests = [], []
    for node in mod.body:
        src = ast.get_source_segment(r["test"], node)
        if isinstance(node, ast.ImportFrom) and node.module == "solution":
            if [a.name for a in node.names] != [fn]: return None
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(n.split(".")[0] not in SAFE_IMPORTS for n in names): return None
            pre.append(src); continue
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            if node.args.args or node.decorator_list: return None   # fixtures or parametrisation
            tests.append((node.name, src)); continue
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.ClassDef)):
            pre.append(src); continue
        if isinstance(node, ast.If) and "__main__" in src or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue   # `if __name__ == "__main__": pytest.main()`, module docstrings
        return None
    if len(tests) < 3: return None
    head = "\n".join(pre)
    block = lambda name, src: (head + "\n\n" if head else "") + src + f"\n\n{name}()"
    return {"id": f"kodcode-{r['question_id']}", "source": "kodcode", "problem": r["question"].strip(), "fn": fn, "signature": decl,
            "visible": [block(*tests[0])], "shown": (head + "\n\n" if head else "") + tests[0][1], "heldout": [block(n, s) for n, s in tests[1:]],
            "difficulty": r["gpt_difficulty"], "subset": r["subset"], "reference": r["solution"]}


def mbpp_record(r):
    tested = re.findall(r"\b([A-Za-z_]\w*)\s*\(", " ".join(r["test_list"]))
    defs = {n.name: n for n in ast.parse(r["code"]).body if isinstance(n, ast.FunctionDef)}
    fn = next((t for t in tested if t in defs), None)
    if fn is None: return None
    sig = r["code"].splitlines()[defs[fn].lineno - 1].strip()
    imports = "\n".join(r["test_imports"])
    vis = [(imports + "\n" if imports else "") + t for t in r["test_list"]]
    return {"id": f"mbppplus-{r['task_id']}", "source": "mbppplus", "problem": r["prompt"].strip(), "fn": fn, "signature": sig,
            "visible": vis, "shown": "\n".join(vis), "heldout": [(imports + "\n" if imports else "") + r["test"]], "reference": r["code"]}


class _Rename(ast.NodeTransformer):
    def __init__(self, fn): self.fn = fn
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.fn, ctx=node.ctx), node) if node.id == "candidate" else node


def humaneval_record(r, orig, k=3):
    """orig: the original HumanEval problem (openai/openai_humaneval) with the same task_id; its first k asserts that call
    the function are the visible tests."""
    fn = r["entry_point"]
    check = next((n for n in ast.parse(orig["test"]).body if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    if check is None: return None
    vis = [ast.unparse(_Rename(fn).visit(n)) for n in check.body
           if isinstance(n, ast.Assert) and any(isinstance(x, ast.Name) and x.id == "candidate" for x in ast.walk(n))][:k]
    if not vis: return None
    node = next((n for n in ast.parse(r["prompt"]).body if isinstance(n, ast.FunctionDef) and n.name == fn), None)
    if node is None: return None
    sig = r["prompt"].splitlines()[node.lineno - 1].strip()
    return {"id": r["task_id"], "source": "humanevalplus", "problem": r["prompt"].strip(), "fn": fn, "signature": sig,
            "visible": vis, "shown": "\n".join(vis), "heldout": [r["test"] + f"\n\ncheck({fn})\n"], "exec_prefix": r["prompt"],
            "reference": r["prompt"] + r["canonical_solution"]}


def solution_code(rec, code):
    """What runs before the tests: the problem's own code where it has some (HumanEval stubs and helpers), then the solution."""
    return rec.get("exec_prefix", "") + "\n\n" + code


def reference_passes(recs, workers=32):
    """Whether each problem's reference solution passes all its visible and held-out tests under this harness."""
    from dcjev.code.sandbox import Executor, pass_count
    ex = Executor(workers)
    vis = ex([(solution_code(r, r["reference"]), r["visible"]) for r in recs])
    held = ex([(solution_code(r, r["reference"]), r["heldout"]) for r in recs], per_test_s=10, total_s=60)
    return [pass_count(v, len(r["visible"])) == len(r["visible"]) and pass_count(h, len(r["heldout"])) == len(r["heldout"])
            for r, v, h in zip(recs, vis, held)]


def build(out: Path):
    """Problems whose reference solution fails its own tests under this harness are left out (and counted)."""
    import numpy as np
    from datasets import load_dataset
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    def dump(name, recs, raw):
        ok = reference_passes(recs)
        keep = [r for r, x in zip(recs, ok) if x]
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[name] = {"source_problems": raw, "fit_protocol": len(recs), "reference_passes": len(keep)}
        print(name, report[name], flush=True)
        return keep
    raw = load_dataset("KodCode/KodCode-Light-RL-10K", split="train", revision=KOD_REV)
    kod = [x for r in raw if (x := kodcode_record(r))]
    report["kodcode"] = {"source_problems": len(raw), "fit_protocol": len(kod)}
    held = set(np.random.default_rng(0).choice(len(kod), 300, replace=False).tolist())
    dump("kodcode_train", [r for i, r in enumerate(kod) if i not in held], None)
    dump("kodcode_val", [r for i, r in enumerate(kod) if i in held], None)
    raw = load_dataset("evalplus/mbppplus", split="test", revision=MBPP_REV)
    dump("mbppplus", [x for r in raw if (x := mbpp_record(r))], len(raw))
    orig = {r["task_id"]: r for r in load_dataset("openai/openai_humaneval", split="test", revision=HE_ORIG_REV)}
    raw = load_dataset("evalplus/humanevalplus", split="test", revision=HE_REV)
    dump("humanevalplus", [x for r in raw if r["task_id"] in orig and (x := humaneval_record(r, orig[r["task_id"]]))], len(raw))
    (out / "README.json").write_text(json.dumps(report, indent=1))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    build(Path(ap.parse_args().out))
