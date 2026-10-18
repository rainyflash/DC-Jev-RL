# Part III-C offline prefix study (raw data and report)

`python -m dcjev.llm.prefix_study gen|score|report --dir <this folder>` (session 3, base policy, no LoRA).

- `roots.jsonl.gz`: 4 solutions per validation problem (the 500 MATH training problems GRPO holds out), sampled from
  Qwen3.5-4B-Base with GRPO's prompt at T = 1 up to 512 tokens; length, finish reason, and correctness of those that
  ended.
- `prefixes.jsonl.gz`: every root cut at 128, 256 and 512 tokens (when it ran that long), with 4 continuations each to
  1024 tokens in all, verified by math-verify. A prefix's Monte-Carlo value is the fraction of its continuations that
  are correct.
- `scores.jsonl.gz`: per prefix (same order), Kev-4B's p(yes) for `on_track` (`../prompts/prefix_prompts.json`) and for
  Part II's `correct` with the prefix as the proposed solution, and the AR judge's p(yes) for `on_track`. The policy's
  own mean token log-probability is `prefix_lp_mean` in `prefixes.jsonl.gz`.
- `prefix_study.json`: per cut, AUROC for a continuation ending correct (pooled and within problem), Spearman
  correlation with the Monte-Carlo value, and the selection experiment on the problems whose 4 roots all run past the
  cut: keep the 2 prefixes a signal ranks highest and compare their mean value with that of all 4 (what random choice
  keeps in expectation). The oracle ranks by 2 of the 4 continuations and is scored on the other 2, so it is limited by
  the same noise as the signals. 95% intervals are bootstrapped over problems. Also v3's threshold rule (prune when
  p(yes) < 0.3).

The seconds in `score_cost.json` were measured while another job used the GPU; `experiments/validation/judge_bench.json`
has the costs measured on an idle GPU.
