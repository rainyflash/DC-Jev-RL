# Part I: PPO on MuJoCo with judge rewards (raw runs)

`runs/<task>/<condition>/s<seed>/` holds one CleanRL-style PPO run (`dcjev/control/ppo.py`, 1M environment steps):

- `final.json`: the final evaluation (20 episodes of the deterministic policy, with the undiscounted environment return), plus the run's wall-clock time and judge usage;
- `log.jsonl.gz`: every record the run logged. That is the config, each training episode, each PPO update, the periodic evaluations (10 episodes every 100k steps), and a 2% sample of 32-step windows (the behaviour summary the judges read, with the environment reward over that window);
- `policy.pt`: the final policy weights.

`results/` is produced from `runs/` by

    python -m dcjev.analysis.part1 --root experiments/part1_control/runs --out experiments/part1_control/results

Status: complete. 165 runs: 7 conditions x 5 tasks x 5 seeds, 3 seeds for `ar_sampled`. A reset of the Colab VM
interrupted 24 runs (`kev_score`, `kev_gated`, `kev_distill`, `ar_gated`). They resumed from their checkpoint at step
301,056, and each dropped the episode in progress (at most 703 steps). Every resume is a `resume` record in the run's
log and is listed in `results/summary.json` (`resumed_runs`).

`results/behaviour.json` (what the final policies do: standing still, flips onto the back, control effort, fingertip
distance, with a random policy as the reference) is produced by

    python -m dcjev.analysis.behaviour --root experiments/part1_control/runs --out experiments/part1_control/results/behaviour.json

It re-runs every final policy on its 20 evaluation episodes, and the returns reproduce `final.json` exactly.

Main results (interquartile mean of the normalised score, 95% bootstrap interval; the paper is
`papers/part1_control/`):

| reward | IQM | note |
|---|---|---|
| `env` (engineered) | 1.05 [0.91, 1.16] | 1 of 5 HalfCheetah runs ends on its back |
| `env_window` | 1.10 [0.89, 1.26] | 3 of 5 HalfCheetah runs end on their back |
| `kev_score` | 0.78 [0.60, 0.91] | |
| `kev_gated` | 1.00 [0.81, 1.09] | no flips; below a random policy on Reacher |
| `kev_distill` | 0.98 [0.82, 1.10] | judge asked about 10% of the windows |
| `ar_gated` | 0.35 [0.28, 0.46] | Hopper and Walker2d policies learn to stand still |
| `ar_sampled` | 0.31 [0.19, 0.45] | |

The numbers in this table are copied from `results/summary.json`, and the paper's numbers are generated from the
results by `dcjev/analysis/paper_numbers.py`.

`results/judge_agreement.json` and `results/judge_agreement_scores.jsonl.gz` (every window's scores) are produced by

    python -m dcjev.analysis.judge_agreement --root experiments/part1_control/runs --out experiments/part1_control/results/judge_agreement.json

with the judge server running (`scripts/run_part1.sh server`); `--from_scores` recomputes the statistics from the saved
scores without it. The committed file was produced on an otherwise idle GPU of a second Colab VM, with the same code,
server and prompts as the training runs. An earlier run on the training VM, without saved scores, gave correlations
within 0.004 of these for Kev-4B's signals and within 0.02 for the AR judge's (largest on Reacher, where its correlations
are near zero and small numerical differences between batches reorder many windows); the sampled AR level, which is
drawn at random, differed by up to 0.03.
