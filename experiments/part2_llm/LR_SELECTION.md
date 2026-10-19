# Learning-rate selection rule

Sweep: `rlvr`, seed 1, 16 problems x 8 completions per step, 60 steps, learning rates 1e-5, 3e-5 and 1e-4
(`scripts/run_part2.sh sweep <lr>`). Each run evaluates the validation set (500 MATH training problems of levels 3-5
held out with seed 12345; greedy, 2048 tokens) at steps 0, 30 and 60.

Rule: the learning rate with the highest mean validation accuracy over steps 30 and 60 is used for every GRPO run of
Parts II and III (all reward conditions, including the baselines). If the best two are within 0.5 points of each other,
the smaller learning rate is used. No test set is evaluated in the sweep.

When: this rule was committed while the 1e-5 and 1e-4 runs were starting. The 3e-5 run had reached step 59; its
validation accuracy at steps 0 and 30 (0.694, 0.730) had been logged and seen.

The 3e-5 run used the first rollout server (one request at a time); the other two used the continuous-batching server
that every later run uses (`experiments/validation/rollout_server_*.json` compares the two).

## Result

`python -m dcjev.analysis.sweep --root experiments/part2_llm/sweep --out experiments/part2_llm/results/lr_sweep.json`

| learning rate | val acc, step 0 | step 30 | step 60 | score (mean of 30 and 60) |
|---|---|---|---|---|
| 1e-5 | 0.692 | 0.738 | 0.738 | 0.738 |
| 3e-5 | 0.694 | 0.730 | 0.744 | 0.737 |
| 1e-4 | 0.698 | 0.748 | 0.678 | 0.713 |

The best two (1e-5 and 3e-5) are within 0.5 points, so the rule picks the smaller one: **1e-5**, used for every GRPO
run of Parts II and III. At 1e-4 the validation accuracy rose by step 30 and then fell below the base model's by step 60.

## Record of the runs

- The first 1e-5 and 1e-4 runs were lost at step 15 of 60, before their step-30 evaluation, when the Colab runtimes
  were reset at about 05:57 UTC on 18 October 2026. Both were rerun from scratch with the same seed and code; the
  reruns are the ones reported. Only the lost runs' step-0 evaluations had been seen.
- The 1e-4 rerun shared its rollout server with the base model's evaluation (`dcjev.llm.eval_base`) during its first
  13 steps, which made those steps slower.
- The step-0 accuracies differ between the runs (0.692 to 0.698) although step 0 is the same base model: greedy
  decoding on vLLM is not bitwise reproducible across server states. This is the size of the evaluation noise.
- `sweep/lr*/` holds each run's log and final record, and for the two reruns the per-problem evaluation outputs
  (`evals/`). The 3e-5 run predates the per-problem outputs. The rollouts of the sweep runs are not kept in the
  repository.
