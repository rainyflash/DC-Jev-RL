# Part II: GRPO on MATH with a judge on the shared backbone (raw runs)

`runs/<condition>/s<seed>/` holds one GRPO run (`dcjev/llm/grpo.py` via `scripts/run_part2.sh run`; 200 steps of 16
problems with 8 completions each, learning rate 1e-5):

- `final.json`: the final evaluation (greedy MATH-500 and GSM8K test; AMC23, AIME24 and AIME25 with 16 samples per
  problem at T = 0.6), with Kev-4B's view of the MATH-500 completions, and the run's wall-clock time;
- `log.jsonl.gz`: every record the run logged: the config, every step (verifier accuracy and reward, Kev-4B's scores of
  64 completions, group statistics, completion lengths, the vLLM/trainer probability mismatch, timings) and the
  periodic greedy evaluations (validation set, MATH-500, a 500-problem GSM8K subset, every 50 steps);
- `final_completions.jsonl.gz`: every completion of the final evaluation, with the verifier's verdict and, on MATH-500,
  Kev-4B's scores;
- `evals/step<N>.jsonl.gz`: every completion of the periodic evaluation at step N.

The training rollouts (`rollouts.jsonl.gz`, about 10 MB per run) are not in the repository.

Conditions: `rlvr` (verifier), `rlvr_kev_add` (verifier + 0.2 Kev-4B P(correct)), `rlvr_kev_gate` (verifier x Kev-4B
P(genuine)), `kev_only` (Kev-4B P(correct), no verifier), `rlvr_ar_add` (verifier + 0.2 AR-judge P(correct)). Seeds:
three for `rlvr` and `rlvr_kev_add`, two for the others.

Other directories:

- `sweep/`: the three learning-rate sweep runs (60 steps, `rlvr`, seed 1); `LR_SELECTION.md`: the rule that chose
  1e-5 on the validation set, and what was known when it was written;
- `base_eval/`: two greedy evaluations of the base model on the final evaluation's sets;
- `judge_study/`: the offline judge study (its own README);
- `prompts/judge_prompts.json`: the judges' questions, fixed before any GRPO run.

`results/` is produced from `runs/` by

    python -m dcjev.analysis.part2 --root experiments/part2_llm/runs --out experiments/part2_llm/results

(`summary.json`: accuracies per condition, paired per-problem differences to `rlvr` with bootstrap intervals over
problems, training statistics, what the final policies write, and the spread of the step-0 evaluations;
`training_curves.pdf`), and `results/lr_sweep.json` by
`python -m dcjev.analysis.sweep --root experiments/part2_llm/sweep --out experiments/part2_llm/results/lr_sweep.json`.
The paper's numbers come from these files through `dcjev/analysis/paper_numbers.py part2`.

Every run used one Colab RTX PRO 6000 (96 GB) shared with other runs; `../RUNTIME_LOG.md` lists the runtimes and resets.
