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
- `evals/step<N>.jsonl.gz`: every completion of the periodic evaluation at step N;
- `rollouts.jsonl.gz` (about 10 MB): every training completion, 128 per step, with its step, problem id, verdict,
  reward, advantage, and the judges' scores where they were computed (Kev-4B for every completion when it is in the
  reward, otherwise for the 64 the log samples; the AR judge for every completion of `rlvr_ar_add`). The papers'
  analyses of the writing style during training read it.

Conditions: `rlvr` (verifier), `rlvr_kev_add` (verifier + 0.2 Kev-4B P(correct)), `rlvr_kev_gate` (verifier x Kev-4B
P(genuine)), `kev_only` (Kev-4B P(correct), no verifier), `rlvr_ar_add` (verifier + 0.2 AR-judge P(correct)). Seeds:
three for `rlvr` and `rlvr_kev_add`, two for the others.

Other directories:

- `sweep/`: the three learning-rate sweep runs (60 steps, `rlvr`, seed 1); `LR_SELECTION.md`: the rule that chose
  1e-5 on the validation set, and what was known when it was written;
- `base_eval/`: two greedy evaluations of the base model on the final evaluation's sets;
- `judge_study/`: the offline judge study (its own README);
- `repeat_probe/`: a test added after the GRPO runs, when the final policies of `kev_only`, `rlvr_kev_gate` and Part
  III-B's `sub_kev` and `sub_kev_recal` turned out to repeat parts of their solutions: does repetition itself raise the
  judges' scores? `dcjev/llm/repeat_probe.py` scores the judge study's base-policy completions as written and with
  repetition added (the last two paragraphs twice more; the whole completion twice), and repeating runs' MATH-500 final
  completions as written and cut after the first answer, with both judges as in GRPO. `scores.jsonl.gz`: every score;
  `repeat_probe.json`: the mean change per edit, with bootstrap intervals over problems. The first version (seed 1 of
  those four conditions) ran on 2026-10-19 on a GPU shared with two GRPO runs. When later runs showed that policies
  trained without the judge in their reward repeat themselves too (`rlvr` seed 3; Part III-C's `tree_kev` seed 2), the
  probe was run again from the pushed code with these two runs added, on 2026-10-19 between 15:27 and 15:44 UTC on an
  otherwise idle Colab RTX PRO 6000 (Kev-4B 10 minutes, the AR judge 6, model loading included). The files here are
  from that second run. Its rows for the four earlier runs and the base policy match the first version's within 0.002
  (GPU kernels are not deterministic). Command:

      python -m dcjev.llm.repeat_probe --study experiments/part2_llm/judge_study \
          --runs experiments/part2_llm/runs/kev_only/s1 experiments/part2_llm/runs/rlvr_kev_gate/s1 \
                 experiments/part3b_labels/runs/sub_kev/s1 experiments/part3b_labels/runs/sub_kev_recal/s1 \
                 experiments/part2_llm/runs/rlvr/s3 experiments/part3c_tree/runs/tree_kev/s2 \
          --out experiments/part2_llm/repeat_probe

- `prompts/judge_prompts.json`: the judges' questions, fixed before any GRPO run.

`results/` is produced from `runs/` by

    python -m dcjev.analysis.part2 --root experiments/part2_llm/runs --out experiments/part2_llm/results

(`summary.json`: accuracies per condition, paired per-problem differences to `rlvr` with bootstrap intervals over
problems, training statistics, what the final policies write, and the spread of the step-0 evaluations;
`training_curves.pdf`), and `results/lr_sweep.json` by
`python -m dcjev.analysis.sweep --root experiments/part2_llm/sweep --out experiments/part2_llm/results/lr_sweep.json`.
The paper's numbers come from these files through `dcjev/analysis/paper_numbers.py part2`.

Every run used one Colab RTX PRO 6000 (96 GB) shared with other runs; `../RUNTIME_LOG.md` lists the runtimes and resets.
