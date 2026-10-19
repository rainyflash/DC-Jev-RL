# Part II offline judge study (raw data and report)

`python -m dcjev.llm.judge_study gen|score|genjudge|report --dir <this folder>`. The stages are described in the module docstring.

- `completions.jsonl.gz`: 4 completions (T = 1, the GRPO prompt and stop strings, at most 1024 tokens) of the base policy
  Qwen3.5-4B-Base for each of 1000 problems (MATH-500 and the 500-problem GSM8K-test subset), verified by math-verify
  (`gen.json`: accuracy 0.609).
- `scores.jsonl.gz`: per completion and per edited copy (`answer_only`, `padded`, `asserted`, `altered`), Kev-4B's
  `correct`, `genuine` and `quality` answers and the AR judge's (Qwen3.5-4B, thinking off, one forward pass) `correct`
  and `quality` answers.
- `genjudge.jsonl.gz`: the AR model reasoning first (thinking on, at most 2048 thinking tokens, T = 0.6), on a fixed
  random 1000 of the completions; p(yes) read after the reasoning.
- `judge_study.json`: the report (AUROC pooled and within problem, ECE, Brier, the shift each edit causes, cost).

Costs: the seconds in `score_cost.json` were measured while other jobs shared the GPU (Kev under heavier load than the
AR judge), so they do not compare the judges. The cost comparison in the paper comes from a separate measurement on an
idle GPU. Generation: session 1; scoring: session 1; generative judge and report: session 2 (all RTX PRO 6000 Blackwell
Server Edition, identical environments).
