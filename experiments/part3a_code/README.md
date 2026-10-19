# Part III-A: multi-turn code repair with visible tests (raw runs)

`runs/<condition>/s<seed>/` holds one GRPO run (`dcjev/code/grpo_code.py` via `scripts/run_part3.sh a`; 100 steps of 16
problems with 8 episodes each, learning rate 1e-5). An episode is up to three attempts at a function: after each attempt
the visible tests run, and the first failing one is fed back. The final attempt is then run against the held-out tests.

- `final.json`: the final greedy evaluation (one episode per problem) on kodcode_val, MBPP+ and HumanEval+: visible pass,
  all-tests pass (visible and held-out), hack rate (visible tests pass, a held-out test fails), first-attempt rates,
  attempts and tokens, and Kev-4B's view of the final programs;
- `log.jsonl.gz`: every record the run logged: the config, every step (the rates above on the training episodes, the
  reward and advantage statistics, Kev-4B's answers, timings) and the periodic greedy evaluations (every 50 steps);
- `final_episodes.jsonl.gz`: every episode of the final evaluation: the attempts, the verdicts and Kev-4B's answers;
- `episodes.jsonl.gz`: every training episode (about 2 MB per run): the attempts, the verdicts, Kev-4B's answers where
  they were computed, the reward and the advantage. `dcjev/analysis/code_hacks.py` reads them.

Conditions (reward of an episode, from its final attempt): `vis` (1 if it passes every visible test), `vis_kev` (`vis`
times Kev-4B's P(yes) for `genuine`, does the code solve the problem rather than special-case the example tests), and
`hidden` (1 if it passes the visible and the held-out tests: an oracle, what `vis` would be with perfect tests). Two
seeds each.

The code tasks are built by `python -m dcjev.code.data --out <dir>` (`scripts/run_part3.sh code_data`) from pinned
revisions of KodCode-Light-RL-10K, MBPP+ and HumanEval+; the module's docstring says which tests are visible.

Other directories:

- `judge_study/`: the offline code-judge study (its own README);
- `prompts/code_prompts.json`: the task template, the number of attempts and the judges' questions, committed before the
  judge study and the GRPO runs and not changed since.

`results/` is produced from `runs/` by

    python -m dcjev.analysis.part3a --root experiments/part3a_code/runs --out experiments/part3a_code/results
    python -m dcjev.analysis.code_hacks --root experiments/part3a_code/runs --data <code data dir> \
        --study experiments/part3a_code/judge_study --out experiments/part3a_code/results/code_hacks.json

(`summary.json`: rates per condition and set, paired per-problem differences to `vis` with bootstrap intervals over
problems, training statistics and the spread of the step-0 evaluations; `training_curves.pdf`; `code_hacks.json`: how
often programs copy a literal of the visible tests that the reference solution does not contain, on the judge study's
programs, during training and in the final evaluation). The paper's numbers come from these files through
`dcjev/analysis/paper_numbers.py part3a`.

Every run used one Colab RTX PRO 6000 (96 GB) shared with other runs; `../RUNTIME_LOG.md` lists the runtimes and resets.
