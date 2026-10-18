# Part III-A offline code judge study (raw data and report)

`python -m dcjev.code.judge_study gen|score|report --dir <this folder>` (session 3).

- `codes.jsonl.gz`: single attempts of the base policy (Qwen3.5-4B-Base, T = 1, 4 per problem) on kodcode_val, MBPP+
  and HumanEval+, run on the visible and the held-out tests; plus, for MBPP+ and HumanEval+, each problem's reference
  solution and, where the visible tests have the form `assert f(args) == value`, a hard-coded lookup of the visible tests
  (kept only if it passes them; `gen.json` counts how many of those also pass the held-out tests).
- `scores.jsonl.gz`: Kev-4B's and the AR judge's p(yes) for `genuine` and `correct` on every program.
- `judge_study.json`: AUROC for passing the held-out tests among the attempts that pass the visible tests, AUROC for
  passing everything over all attempts, and reference against hard-coded programs.

The seconds in `score_cost.json` were measured with other jobs on the GPU and do not compare the judges.
