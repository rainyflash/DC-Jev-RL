# Correction notice: the results of versions 1.0–3.0 are withdrawn

*October 2026. [中文版见下方](#更正声明第-10-30-版的结果全部撤回)*

All results reported in versions 1.0 to 3.0 of the five papers in this repository are withdrawn. That covers the papers, figures, leaderboards, README tables and the Zenodo and Hugging Face records built from them:

| Part | v1–v3 title (short) | Zenodo record |
|---|---|---|
| I | DC-Jev-RL (continuous control) | [10.5281/zenodo.23439711](https://doi.org/10.5281/zenodo.23439711) |
| II | DC-Jev-GRPO (LLM reasoning) | [10.5281/zenodo.23441431](https://doi.org/10.5281/zenodo.23441431) |
| III-A | Agentic-Jev-GRPO (multi-turn code and tool use) | [10.5281/zenodo.23442087](https://doi.org/10.5281/zenodo.23442087) |
| III-B | CoEvo-Jev-GRPO (judge–policy co-evolution) | [10.5281/zenodo.23442137](https://doi.org/10.5281/zenodo.23442137) |
| III-C | Tree-Jev-GRPO (prefix trees, theorem proving) | [10.5281/zenodo.23442169](https://doi.org/10.5281/zenodo.23442169) |

## What was wrong

An audit of this repository in October 2026 found that the reported results were not produced by running the experiments the papers describe:

- **Part I.** During PPO training, the judge was not Kev-4B. It was a hand-written function of the simulator state: logistic curves of forward velocity, posture and control effort (`kev4b_calibrated_response` in `withdrawn/jev_rl/controlled_benchmark.py`, `kev4b_system_one_rubric` in `withdrawn/jev_rl/leaderboard_suite.py`). The code calls this function "Kev-4B's calibrated response", and its docstring says it was fitted to Kev-4B's answers on three hand-made probe windows, but the model was never queried during training. The normalised-score leaderboard and the "100% win rate over the human oracle" were built on those runs and on constants in the scripts.
- **Part II.** The leaderboard values (accuracies, losses, dead-zone and hack rates, and the "observed pathology" descriptions) are literals in `withdrawn/jev_llm_rl/llm_leaderboard_suite.py`. They are not outputs of training runs. The Colab script `withdrawn/jev_llm_rl/blackwell_experiment.py` did load the model, but it did not measure the reported memory saving. It set the "two separate models" memory to 2.04 × the measured single-backbone memory and the "two adapters" memory to that measurement + 0.12 GB.
- **Parts III-A, III-B and III-C.** The benchmark suites (`withdrawn/jev_agentic_rl/`, `withdrawn/jev_coevo_rl/`, `withdrawn/jev_tree_grpo/`) contain no code that loads, trains or evaluates a model. Every accuracy, learning curve and ablation in these papers is a number written into those scripts. None of the benchmarks named in them was run: BFCL-v3, MBPP+, HumanEval+, SWE-bench-Verified-Mini, MATH-500, AIME, AMC, OlympiadBench, MiniF2F-Lean4, ProofNet, GPQA-Diamond.
- **All parts.** The memory and latency figures for running Kev-4B on the same backbone as the policy were not measured. Examples are "8.06 GB vs 16.20 GB", a "0.012 ms" adapter switch, "14.80 ms per node" and "84.6% KV-cache reuse". The "SOTA" and "#1" claims have no basis.

## What replaces them

Version 4, on this branch, redoes the work as measured experiments, with a smaller scope where the original one was not feasible. The rules are:

- every number in the v4 papers is computed by the scripts in `dcjev/analysis/` from logs committed under `experiments/`, together with the code, seeds and configuration;
- baselines get the same tuning as the proposed methods, and tuning uses held-out validation data, never a test set;
- negative and null results are reported as such.

Several original claims turned out to be testable and are tested directly. One example is the memory saved by sharing one Qwen3.5-4B backbone between the policy and Kev-4B, which v4 measures. Others turned out not to hold, or could not be tested at this scale, and the v4 papers say which.

The v1–v3 files are kept under [`withdrawn/`](withdrawn/) for the record. Their only changes are this notice and the redaction of a credential.

---

# 更正声明：第 1.0–3.0 版的结果全部撤回

*2026 年 10 月*

本仓库五篇论文第 1.0 至 3.0 版中报告的全部结果予以撤回。撤回范围包括论文、图表、排行榜、README 中的表格，以及据此发布的 Zenodo 与 Hugging Face 记录（DOI 见上表）。

## 问题所在

2026 年 10 月对本仓库的审查发现，这些结果并非由论文所描述的实验实际运行得出：

- **第一部分。** PPO 训练中使用的“评审模型”并不是 Kev-4B，而是一个根据仿真器状态手工编写的函数，即前进速度、姿态和控制量的 logistic 曲线（`withdrawn/jev_rl/controlled_benchmark.py` 中的 `kev4b_calibrated_response`，`withdrawn/jev_rl/leaderboard_suite.py` 中的 `kev4b_system_one_rubric`）。代码中把它称为“Kev-4B 的校准响应”，其文档字符串称它是按 Kev-4B 在三个手工构造的探测窗口上的回答拟合的，但训练过程中从未调用过该模型。归一化得分排行榜和“对人类基准 100% 胜率”都建立在这些运行和脚本中的常数之上。
- **第二部分。** 排行榜中的数值（准确率、损失、死区率、作弊率以及“观察到的病理现象”描述）是直接写在 `withdrawn/jev_llm_rl/llm_leaderboard_suite.py` 中的常量，并非训练运行的输出。Colab 脚本 `withdrawn/jev_llm_rl/blackwell_experiment.py` 确实加载了模型，但它没有测量所报告的显存节省：“两个独立模型”的显存被设为实测单主干显存的 2.04 倍，“两个适配器”的显存被设为该实测值加 0.12 GB。
- **第三部分 A、B、C。** 相应的基准测试脚本（`withdrawn/jev_agentic_rl/`、`withdrawn/jev_coevo_rl/`、`withdrawn/jev_tree_grpo/`）中没有任何加载、训练或评测模型的代码。这三篇论文中的每一个准确率、学习曲线和消融结果都是写在脚本里的数字。文中提到的基准一个都没有实际运行过：BFCL-v3、MBPP+、HumanEval+、SWE-bench-Verified-Mini、MATH-500、AIME、AMC、OlympiadBench、MiniF2F-Lean4、ProofNet、GPQA-Diamond。
- **所有部分。** 关于 Kev-4B 与策略共享同一主干时的显存和延迟数字均未经测量，例如“8.06 GB 对 16.20 GB”、“0.012 ms”的适配器切换、“每节点 14.80 ms”、“84.6% KV 缓存复用”。“SOTA”和“第一名”的说法没有依据。

## 替代内容

本分支上的第 4 版以实际测量的实验重做这项工作。原定范围无法实现的部分相应缩小。规则如下：

- 第 4 版论文中的每一个数字，都由 `dcjev/analysis/` 中的脚本从提交在 `experiments/` 下的日志计算得出，代码、随机种子和配置一并提交；
- 基线方法与所提方法享有同等的调参，调参只使用留出的验证集，从不使用测试集；
- 负面结果和无显著差异的结果如实报告。

一些原有论断是可以检验的，第 4 版直接检验了它们。例如 Kev-4B 与策略共享同一个 Qwen3.5-4B 主干时节省的显存，第 4 版做了实测。另一些论断经检验不成立，或在当前规模下无法检验，第 4 版论文都会注明是哪些。

第 1–3 版的文件保留在 [`withdrawn/`](withdrawn/) 目录中以备查阅，唯一的改动是加上本声明和删去一处凭据。
