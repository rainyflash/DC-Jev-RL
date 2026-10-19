# Part III-C: tree rollouts with roots chosen by a judge (raw runs)

`runs/<condition>/s<seed>/` holds one GRPO run (`dcjev/llm/grpo.py` via `scripts/run_part3.sh c`), with Part II's
`rlvr` setup (200 steps of 16 problems, verifier reward, learning rate 1e-5) except for how each group of 8 completions
is sampled (`dcjev/llm/tree.py`): 4 roots of 128 tokens are sampled per problem, 2 of them are continued with 4 samples
each from the exact token prefix, and the 8 leaves form the group.

- `tree_kev`: the 2 roots with the highest Kev-4B P(yes) for `on_track` (`prompts/prefix_prompts.json`) are continued;
- `tree_random`: 2 roots chosen at random (in distribution the same as sampling 2 roots only).

Two seeds each. The files are those of a Part II run (`../part2_llm/README.md`): `final.json`, `log.jsonl.gz`,
`final_completions.jsonl.gz` and `evals/step<N>.jsonl.gz`, and in addition `rollouts.jsonl.gz`, every training
completion (leaf) with its step, problem id, root, verdict, reward and advantage; the paper's analysis of repetition
during training reads it. Every step's log record also has the tree statistics: how
many roots had already ended, Kev-4B's `on_track` scores (mean, mean of the kept roots, AUROC for the leaves' verdicts,
pooled and within problem) and the tokens generated. Every tenth step the roots that were not kept are continued as
well, for the log only: `audit_selection_gain` is the mean value of the kept roots minus that of all 4, and
`audit_best_possible_gain` the same for the best 2 in hindsight.

Other directories:

- `prefix_study/`: the offline study of judges on unfinished solutions (its own README);
- `prompts/prefix_prompts.json`: the `on_track` question, committed before the prefix study ran and not changed since.

`results/` is produced from these runs and Part II's `rlvr` by

    python -m dcjev.analysis.part2 --root experiments/part3c_tree/runs --root experiments/part2_llm/runs \
        --conds rlvr,tree_random,tree_kev --baseline rlvr,tree_random --out experiments/part3c_tree/results

and `results/summary_step100.json` by the same command with `--step 100` (the periodic evaluations at step 100, written
to another folder and copied here). The paper's numbers come from these files and `prefix_study/` through
`dcjev/analysis/paper_numbers.py part3c`.

Every run used one Colab RTX PRO 6000 (96 GB) shared with other runs; `../RUNTIME_LOG.md` lists the runtimes and resets.
