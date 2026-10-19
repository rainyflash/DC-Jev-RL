# DC-Jev-RL, version 4

*[中文版见下方](#dc-jev-rl第-4-版)*

> **Correction.** The results of versions 1.0–3.0 of the five papers in this repository were not produced by the
> experiments they describe, and they are withdrawn. [CORRECTION.md](CORRECTION.md) lists what was wrong. Version 4
> redoes the work as measured experiments.

## Status (19 October 2026)

All five papers of version 4 are complete, in English and Chinese. Every table and number in them is generated from
the logs under `experiments/` by `dcjev/analysis/paper_numbers.py`; none is typed in by hand. The experiments ran on
three Colab GPUs (RTX PRO 6000 Blackwell, 96 GB). The papers report what the runs showed, including where a judge did
not help, and say where a decision was made after seeing results.

| Part | Question | What we found | Paper |
|---|---|---|---|
| I | Do Kev-4B's judgments of short motion windows make a useful PPO reward on five MuJoCo tasks, compared with the environment reward and with an autoregressive judge of the same size? | In 165 PPO runs (seven reward sources, five tasks), Kev-4B's gated reward reaches an interquartile mean normalised score of 1.00 [0.81, 1.09], the engineered reward 1.05, the AR judge's 0.35. On Reacher every judge's reward ends below a random policy, and the AR judge's reward teaches Hopper and Walker2d to stand still. | [EN](papers/part1_control/main_en.pdf), [中文](papers/part1_control/main_zh.pdf) |
| II | What does sharing one Qwen3.5-4B backbone between the GRPO policy and Kev-4B save, and does Kev-4B's judgment add anything to the verifier reward on MATH? | With one shared backbone the trainer needs 48% less GPU memory after loading, and the judge's probabilities stay within 0.015 of fp32. The verifier alone takes MATH-500 from 76.5% to 82.5%. With Kev-4B's score added to the reward, or gating it, one seed per condition ends far ahead and the others near the verifier-only runs, so the gain cannot be attributed to the judge. Kev-4B alone as the reward: 70.7%. | [EN](papers/part2_grpo/main_en.pdf), [中文](papers/part2_grpo/main_zh.pdf) |
| III-A | How much reward hacking is there when multi-turn code repair is rewarded by visible tests and graded by held-out ones, and does a judge reduce it? | No sign that the visible tests were exploited: the share of problems solved rose (KodCode validation 65.5% to 86.6%) and the share of visible-test passes that are wrong fell by about half on two of the three sets. The judge factor changed nothing beyond the variation between seeds. | [EN](papers/part3a_code/main_en.pdf), [中文](papers/part3a_code/main_zh.pdf) |
| III-B | When the verifier labels only 2 of 8 samples per prompt, can a frozen, a recalibrated or a scratch judge head fill in the rest? | The frozen Kev-4B head ranks the unlabelled samples well (AUROC 0.91), but its policies learned to repeat themselves; on MATH-500 they end −2.3 points [−4.4, −0.1] against dropping those samples. A head trained from scratch ends +4.1 [1.6, 6.6] against dropping them, with both seeds switched to drafting first (below). | [EN](papers/part3b_labels/main_en.pdf), [中文](papers/part3b_labels/main_zh.pdf) |
| III-C | Can a judge choose the prefixes to branch from in tree rollouts? | Kev-4B's choice of roots raised the share of correct leaves (+1.4 points over keeping all four roots), and the final policies are as accurate as with random roots (MATH-500 +0.7 [−0.8, 2.3]). The tree runs' lead over flat GRPO comes mostly from the 2 seeds that switched to drafting first. Trees generate 7%–12% fewer tokens than flat sampling of completions of the same lengths. | [EN](papers/part3c_tree/main_en.pdf), [中文](papers/part3c_tree/main_zh.pdf) |

Intervals are 95% bootstrap intervals: over runs, stratified by task, in Part I; over problems in the other parts,
where they do not include the variation between seeds, which in these runs is often larger than the differences
between conditions.

### Across the GRPO papers

- **Drafting first.** Of the 26 GRPO runs of Parts II, III-B and III-C, the 6 whose final policies open
  most solutions with an informal draft in plain text are the 6 best on AIME24 (30.4–34.6%, every
  other run at most 24.4%). They come from 5 conditions, 2 of them rewarded by the verifier alone. We
  found the switch after the runs had finished and defined its measure after reading the completions; the verdicts we
  logged do not show what set it off (Part II, §8.3).
- **Repetition.** Kev-4B pays for solutions that solve the problem again after giving the answer. 6 of the
  11 runs with Kev-4B's score in the reward ended repeating themselves, against 2 of the 15 others;
  repeating did not make a policy more accurate (Part II, §8.3–8.4).

The policies' weights (LoRA adapters) are not kept in the repository; their logs, evaluation outputs and every
training completion are.

## Layout

- `dcjev/`: the code. Judges (Kev-4B and an autoregressive Qwen3.5-4B judge), PPO, the GRPO trainers, the offline
  studies and the analysis.
- `scripts/`: environment setup (`setup_colab.sh`, with the exact package lists in `env/`), one run script per part,
  the GRPO job lists, and `build_papers.sh`.
- `experiments/`: logs, prompts, configurations and results of every run. `RUNTIME_LOG.md` records the runtime resets
  and what they cost.
- `papers/`: LaTeX sources and PDFs of the five papers, in English and Chinese. Their numbers and tables come from
  `dcjev/analysis/paper_numbers.py`, never by hand.
- `withdrawn/`: the files of versions 1.0–3.0, kept for the record.

## Reproducing

1. `bash scripts/setup_colab.sh` on a Linux machine with a recent NVIDIA GPU: both Python environments at the pinned
   versions, the pinned model snapshots and the data.
2. `bash scripts/run_validation.sh`: the checks in `experiments/validation/`.
3. Part I: `bash scripts/run_part1.sh baselines`, then `server` and `judges`.
4. Parts II and III: a rollout server (`bash scripts/run_part2.sh rollout`), then the job lists made by
   `bash scripts/make_jobs.sh 1e-5`, run with `python -m dcjev.queue`.
5. Papers: `python -m dcjev.analysis.paper_numbers <part> --out papers/<paper>`, then `bash scripts/build_papers.sh papers`.

## License

This repository is licensed under the [Apache License 2.0](LICENSE). Two files adapt code released under the MIT
licence: `dcjev/control/ppo.py` (from CleanRL) and the sandbox guard in `dcjev/code/sandbox.py` (from OpenAI's
human-eval); their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Benchmark problems and tests that
appear in `experiments/` keep the licences of their datasets.

---

# DC-Jev-RL，第 4 版

> **更正。** 本仓库五篇论文第 1.0–3.0 版的结果并非由文中所述实验产生，现已全部撤回，错误之处见
> [CORRECTION.md](CORRECTION.md)。第 4 版以实际测量的实验重做这项工作。

## 进度（2026 年 10 月 19 日）

第 4 版的五篇论文已全部完成，各有英文版和中文版。论文中的每个表格和数字都由 `dcjev/analysis/paper_numbers.py` 从
`experiments/` 下的日志生成，没有手填。实验在三块 Colab GPU（RTX PRO 6000 Blackwell，96 GB）上运行。论文如实报告 run 的结果，包括评判模型没有帮助的地方，也说明了哪些决定是在看到结果之后做出的。

| 部分 | 问题 | 结果 | 论文 |
|---|---|---|---|
| 第一部分 | 在五个 MuJoCo 任务上，Kev-4B 对短时运动窗口的判断能否作为有用的 PPO 奖励？与环境奖励和同等规模的自回归评判模型相比如何？ | 在 165 个 PPO run（七种奖励来源、五个任务）中，Kev-4B 带门控的奖励的归一化得分四分位均值为 1.00 [0.81, 1.09]，工程化奖励为 1.05，自回归评判模型为 0.35。在 Reacher 上所有评判奖励最终都不如随机策略；自回归评判模型的奖励让 Hopper 和 Walker2d 学会了站着不动。 | [EN](papers/part1_control/main_en.pdf)、[中文](papers/part1_control/main_zh.pdf) |
| 第二部分 | GRPO 策略与 Kev-4B 共用一个 Qwen3.5-4B 骨干能节省什么？在 MATH 上，Kev-4B 的判断能否为验证器奖励带来增益？ | 共用一份骨干时，训练器加载后少用 48% 的显存，评判模型的概率与 fp32 相差不超过 0.015。只用验证器就把 MATH-500 从 76.5% 提高到 82.5%。把 Kev-4B 的分数加入奖励或用它给验证器做门控时，每个条件有一个种子遥遥领先，其余与只用验证器的 run 相近，因此无法把增益归于评判模型。只用 Kev-4B 作奖励：70.7%。 | [EN](papers/part2_grpo/main_en.pdf)、[中文](papers/part2_grpo/main_zh.pdf) |
| 第三部分 A | 多轮代码修复以可见测试为奖励、以留出测试评分时，奖励投机有多严重？评判模型能否减少它？ | 没有看到可见测试被钻空子的迹象：解出题目的比例上升（KodCode 验证集从 65.5% 到 86.6%），通过可见测试的程序中错误者的比例在三个集合中的两个上降低了约一半。评判因子带来的变化没有超出种子之间的差异。 | [EN](papers/part3a_code/main_en.pdf)、[中文](papers/part3a_code/main_zh.pdf) |
| 第三部分 B | 验证器只为每个提示的 8 个样本中的 2 个打标签时，冻结、重新校准或从头训练的评判输出头能否补全其余标签？ | 冻结的 Kev-4B 输出头对未标注样本排序良好（AUROC 0.91），但它的策略学会了重复自己；在 MATH-500 上，相对于丢掉这些样本，它们最终为 −2.3 个百分点 [−4.4, −0.1]。从头训练的输出头为 +4.1 [1.6, 6.6]，它的两个种子都改成了先写草稿（见下）。 | [EN](papers/part3b_labels/main_en.pdf)、[中文](papers/part3b_labels/main_zh.pdf) |
| 第三部分 C | 在树状 rollout 中，评判模型能否选择从哪些前缀分支？ | Kev-4B 选根让正确叶子的比例比保留全部四个根时变化 +1.4 个百分点，最终策略与随机选根一样准确（MATH-500 +0.7 [−0.8, 2.3]）。树状 run 相对扁平 GRPO 的领先主要来自改成先写草稿的 2 个种子。与生成同样长度解答的扁平采样相比，树少生成 7%–12% 的词元。 | [EN](papers/part3c_tree/main_en.pdf)、[中文](papers/part3c_tree/main_zh.pdf) |

区间是 95% bootstrap 区间：第一部分按任务分层对 run 重采样，其他部分对题目重采样；后者不包含种子之间的差异，而在这些 run 中，种子之间的差异常常大于条件之间的差异。

### 三篇 GRPO 论文的共同发现

- **先写草稿。** 在第二部分、第三部分 B 和 C 的 26 个 GRPO run 中，最终策略的多数解答以纯文本非正式草稿开头的 6 个 run，正是 AIME24 上最好的 6 个（30.4–34.6%，其他 run 至多 24.4%）。它们分属 5 个条件，其中 2 个的奖励只有验证器。这一切换是在 run 结束后才发现的，度量也是在读过解答之后定义的；我们记录的验证结果看不出它因何而起（第二部分 §8.3）。
- **重复。** Kev-4B 会为给出答案之后把题目再解一遍的解答付分。奖励中含有 Kev-4B 分数的 11 个 run 中有 6 个最终重复自己，其他 15 个 run 中只有 2 个；重复并没有让策略更准确（第二部分 §8.3–8.4）。

仓库中没有保存策略的权重（LoRA 适配器）；它们的日志、评估输出和每个训练解答都在仓库中。

## 目录

- `dcjev/`：代码。评判模型（Kev-4B 和自回归的 Qwen3.5-4B 评判模型）、PPO、GRPO 训练器、离线研究和分析。
- `scripts/`：环境搭建（`setup_colab.sh`，确切的包版本列表在 `env/` 中）、各部分的运行脚本、GRPO 任务列表和 `build_papers.sh`。
- `experiments/`：每个 run 的日志、提示、配置和结果。`RUNTIME_LOG.md` 记录了运行时重置及其损失。
- `papers/`：五篇论文的 LaTeX 源文件和 PDF（英文和中文）。其中的数字和表格全部由 `dcjev/analysis/paper_numbers.py` 生成，没有手填。
- `withdrawn/`：第 1.0–3.0 版的文件，留档备查。

## 复现

1. 在装有较新 NVIDIA GPU 的 Linux 机器上运行 `bash scripts/setup_colab.sh`：按固定版本安装两个 Python 环境，下载固定版本的模型快照和数据。
2. `bash scripts/run_validation.sh`：`experiments/validation/` 中的各项检查。
3. 第一部分：`bash scripts/run_part1.sh baselines`，然后是 `server` 和 `judges`。
4. 第二、三部分：先启动 rollout 服务（`bash scripts/run_part2.sh rollout`），再用 `python -m dcjev.queue` 运行 `bash scripts/make_jobs.sh 1e-5` 生成的任务列表。
5. 论文：`python -m dcjev.analysis.paper_numbers <part> --out papers/<paper>`，然后 `bash scripts/build_papers.sh papers`。

## 许可证

本仓库采用 [Apache License 2.0](LICENSE) 许可。有两个文件改编自以 MIT 许可发布的代码：`dcjev/control/ppo.py`（来自 CleanRL）和 `dcjev/code/sandbox.py` 中的沙箱防护（来自 OpenAI 的 human-eval），其许可声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。`experiments/` 中出现的基准题目和测试仍遵循各自数据集的许可。
