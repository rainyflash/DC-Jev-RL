# Part III-B: GRPO on MATH when only a quarter of the completions are labelled (raw runs)

`runs/<condition>/s<seed>/` holds one GRPO run (`dcjev/llm/grpo.py` via `scripts/run_part3.sh b`), with Part II's setup
(200 steps of 16 problems with 8 completions each, learning rate 1e-5) except that the verifier labels only 2 of each
group's 8 completions, chosen at random (`--label_frac 0.25`). The files are those of a Part II run
(`../part2_llm/README.md`): `final.json`, `log.jsonl.gz`, `final_completions.jsonl.gz`, `evals/step<N>.jsonl.gz` and
`rollouts.jsonl.gz`, whose records also say whether a completion was labelled and whether it was used in the update.
The verifier still checks the unlabelled completions, for the log only. In the three conditions that fill in a reward
for them, every step's log record says how good the fill-in was: its AUROC and calibration error for the verifier's
verdict, and the frozen Kev-4B's AUROC on the same completions.

Conditions (what the 6 unlabelled completions of a group get):

- `rlvr_sub`: nothing; they are left out of the update, and the advantages come from the 2 labelled ones;
- `sub_kev`: Kev-4B's P(correct), frozen as released;
- `sub_kev_recal`: Kev-4B's P(correct) after its pointer head is refitted online on all labelled completions so far
  (cross-entropy to the verifier's label, anchored to the released head);
- `sub_probe`: a head of the same shape trained from scratch on the same labels and Kev-4B's features.

Two seeds each. Part II's `rlvr` (every completion labelled) is the reference with the full label budget.

`results/` is produced from these runs and Part II's by

    python -m dcjev.analysis.part2 --root experiments/part3b_labels/runs --root experiments/part2_llm/runs \
        --conds rlvr,rlvr_sub,sub_kev,sub_kev_recal,sub_probe --baseline rlvr_sub --out experiments/part3b_labels/results

(`summary.json`, `training_curves.pdf`; the paired differences are to `rlvr_sub`). The paper's numbers come from these
files through `dcjev/analysis/paper_numbers.py part3b`.

Every run used one Colab RTX PRO 6000 (96 GB) shared with other runs; `../RUNTIME_LOG.md` lists the runtimes and resets.
