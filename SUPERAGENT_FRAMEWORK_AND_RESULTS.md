# SuperAgent 框架、实验方案与结果

记录日期为 2026 年 11 月 22 日。准确率按验证集与测试集分别列出。「—」表示相应运行无该结果。

## 1. 初版 Agent 框架

任务输入 → 数据画像与知识检索 → LLM 规划 → 生成 Python 首稿 → 子进程执行与自动调试 → 按验证准确率迭代选优 → FINAL 阶段生成测试预测与交付文件。

| 部分 | 项目文件 | 功能 |
|---|---|---|
| 运行入口 | [run_agent.py](run_agent.py)、[main.py](superagent/main.py) | 接收任务目录、模型、轮数、指标和预算，启动原始 Agent。现有入口以 `--backend legacy` 保留该模式。 |
| 运行配置 | [config.py](superagent/config.py) | 管理默认模型、价格、预算、执行超时、最大迭代和调试次数。原始默认值为 ¥20 预算、首稿后最多 5 轮改进、单脚本 600 秒超时。 |
| 数据画像 | [profiler.py](superagent/profiler.py) | 扫描传入任务目录，提取文件、表格字段、形状、类型及部分统计与样例。 |
| 知识模块 | [knowledge.py](superagent/knowledge.py)、[static_recipes.md](kb/static_recipes.md) | 载入通用配方，按任务模态和规模检索以往经验，并提供运行后经验写入接口。 |
| LLM 客户端 | [llm.py](superagent/llm.py) | 发送模型请求、解析计划或代码的 JSON、重试并记录请求用量。 |
| 提示协议 | [prompts.py](superagent/prompts.py) | 定义规划、首稿、改进和调试提示，约定 `solution.py` 的输出、`VAL_METRIC` 和 `FINAL=1` 行为。 |
| 脚本执行 | [sandbox.py](superagent/sandbox.py) | 检查 Python 语法，以 `python -I` 子进程运行脚本，捕获 stdout、stderr 和超时状态。 |
| 编排与选优 | [agent.py](superagent/agent.py) | 生成首稿并执行，失败时迭代调试。随后从当前最佳代码生成改进方案，读取脚本输出的 `VAL_METRIC`，只在分数提高时替换最佳代码。连续两轮未提升时停止改进。 |
| 最终评估与输出 | [agent.py](superagent/agent.py)、[report.py](superagent/report.py) | 原始完整入口以 `FINAL=1` 执行选中的脚本，收集产物、运行摘要及费用日志。报告模块汇总运行记录。 |
| 费用记录 | [ledger.py](superagent/ledger.py) | 按配置的价格记录每次调用的输入与输出 token、费用和累计费用，并在请求前检查预算。 |

## 2. 任务与已有手工基线

| 任务 | 输入与预测目标 | 训练集 | 验证集 | 测试集 | 手工基线 | 验证准确率 | 测试准确率 |
|---|---|---:|---:|---:|---|---:|---:|
| 优惠券接受预测 | NPY 表格数据，预测是否接受优惠券 | 8,117 | 2,030 | 2,537 | 独热编码 + HistGradientBoosting | 75.96% | — |
| 标题 CTR | CSV 标题对，预测哪条标题的实际 CTR 更高 | 8,301 | 940 | 3,232 | 词与字符 TF-IDF 差值 + Logistic Regression | 68.62% | — |

手工基线记录见 [改进路线图](SUPERAGENT_IMPROVEMENTS.md)。

## 3. 初版框架的校内 API 试跑

[run_campus_trial.py](run_campus_trial.py) 沿用初版的代码生成、调试和选优流程，校内调用模型为 `DeepSeek-V4.1-Flash`。优惠券试跑使用训练集和验证集，运行首稿和 1 轮改进，预算为 ¥1。

| 轮次 | 实际方案 | 验证准确率 | 测试准确率 | 记录 |
|---|---|---:|---:|---|
| 首版 | 独热编码 Logistic Regression 和 HistGradientBoosting 比较，保留 HistGradientBoosting，阈值 0.48 | 74.78% | — | [候选代码](runs/task1_campus_trial/run_20261122_132426/workspace/best_solution.py) |
| 改进轮 | 对可用模型和阈值进行搜索，本轮选中的组合为单个 HistGradientBoosting，阈值 0.56 | 74.43% | — | [候选代码](runs/task1_campus_trial/run_20261122_132426/workspace/solution.py) |

该次运行的最佳验证准确率为 **74.78%**。成功调用 **5 次**，输入 **26,522** token，输出 **20,284** token，按脚本配置费率折算 **¥0.2153**。见 [运行摘要](runs/task1_campus_trial/run_20261122_132426/logs/run_summary.json)。

| 校内 API 试跑记录 | 运行状态 | 成功调用 | 输入 token | 输出 token | 脚本折算费用 | 验证准确率 |
|---|---|---:|---:|---:|---:|---:|
| [run_20261122_121755](runs/task1_campus_trial/run_20261122_121755/logs/run_summary.json) | 未完成 | 0 | 0 | 0 | ¥0.0000 | — |
| [run_20261122_122119](runs/task1_campus_trial/run_20261122_122119/logs/run_summary.json) | 未完成 | 0 | 0 | 0 | ¥0.0000 | — |
| [run_20261122_123258](runs/task1_campus_trial/run_20261122_123258/logs/run_summary.json) | 未完成 | 2 | 11,784 | 9,996 | ¥0.1035 | — |
| [run_20261122_132426](runs/task1_campus_trial/run_20261122_132426/logs/run_summary.json) | 完成 | 5 | 26,522 | 20,284 | ¥0.2153 | 74.78% |

## 4. 当前框架与已运行方案

训练集和验证集数据画像 → 从已注册的模型及参数中选择候选 → 逐个训练并保存候选 → 框架根据验证标签计算准确率 → 冻结候选 → 使用训练集与验证集重训 → 独立测试集评分。

| 部分 | 项目文件 | 功能 |
|---|---|---|
| 数据读取与评分 | [task_data.py](superagent/task_data.py) | 按 NPY 或 CSV 协议读取开发划分，生成数据画像，检查预测和标签后计算准确率。测试读取接口供最终评估使用。 |
| 分类工具 | [classifiers.py](superagent/classifiers.py) | 提供表格、单文本和成对文本的训练管线及可用参数选项，按依赖和数据特征登记候选。 |
| 候选选择 | [experiment.py](superagent/experiment.py)、[campus_policy.py](superagent/campus_policy.py) | 离线后端按固定规则选候选。校内 API 后端向 LLM 提交数据画像、可选配置和已有验证结果，解析返回的候选 ID。 |
| 执行、记录与冻结 | [experiment.py](superagent/experiment.py) | 按候选分别保存配置、模型、验证预测、验证分数和耗时。对照流程按验证准确率选最终方案，同分时取单起点。冻结配置后重训并保存测试预测与分数。 |

四次离线搜索使用随机种子 42，每次执行 5 个候选。单起点先执行 1 个候选，再细化当前领先的模型族。多起点先执行 3 个模型族，再从排名靠前的两个模型族中细化。四次离线搜索的 API 调用和费用均为 0。下表只在验证阶段选中的一次运行中列出最终测试分数。

| 任务 | 搜索方案 | 完成候选 | 最佳模型 | 最佳验证准确率 | 搜索耗时 | 最终测试准确率 |
|---|---|---:|---|---:|---:|---:|
| 优惠券 | [单起点](runs/structured_comparison/coupon_single/run_20261122_140821_e79a8d/logs/run_summary.json) | 5 | 独热编码 + HGB，63 个最大叶节点 | 76.70% | 24.89 秒 | [76.70%](runs/structured_comparison/coupon_single/run_20261122_140821_e79a8d/final/result.json) |
| 优惠券 | [多起点](runs/structured_comparison/coupon_multi/run_20261122_140821_6f439d/logs/run_summary.json) | 5 | 独热编码 + HGB，63 个最大叶节点 | 76.70% | 40.39 秒 | — |
| 标题 CTR | [单起点](runs/structured_comparison/headlines_single/run_20261122_140821_da4f33/logs/run_summary.json) | 5 | 词 + 字符 TF-IDF 差值 + LR，C=1 | 70.53% | 39.08 秒 | [69.74%](runs/structured_comparison/headlines_single/run_20261122_140821_da4f33/final/result.json) |
| 标题 CTR | [多起点](runs/structured_comparison/headlines_multi/run_20261122_140821_8e78e8/logs/run_summary.json) | 5 | 词 + 字符 TF-IDF 差值 + LR，C=1 | 70.53% | 36.50 秒 | — |

### 优惠券的逐候选验证结果

| 顺序 | 单起点方案 | 验证准确率 | 多起点方案 | 验证准确率 |
|---|---|---:|---|---:|
| 1 | 独热编码 + HGB，31 个最大叶节点 | 75.96% | 独热编码 + HGB，31 个最大叶节点 | 75.96% |
| 2 | 独热编码 + HGB，63 个最大叶节点 | 76.70% | 独热编码 + ExtraTrees，最小叶样本数 2 | 75.91% |
| 3 | 独热编码 + HGB，最小叶样本数 12 | 76.26% | 独热编码 + Logistic Regression | 66.40% |
| 4 | 独热编码 + ExtraTrees，最小叶样本数 2 | 75.91% | 独热编码 + HGB，63 个最大叶节点 | 76.70% |
| 5 | 独热编码 + Logistic Regression | 66.40% | 独热编码 + ExtraTrees，最小叶样本数 1 | 74.58% |

### 标题 CTR的逐候选验证结果

| 顺序 | 单起点方案 | 验证准确率 | 多起点方案 | 验证准确率 |
|---|---|---:|---|---:|
| 1 | 词 + 字符 TF-IDF 差值 + LR，C=1 | 70.53% | 词 + 字符 TF-IDF 差值 + LR，C=1 | 70.53% |
| 2 | 词 + 字符 TF-IDF 差值 + LR，C=3 | 68.94% | 词 TF-IDF 差值 + LR，C=1 | 68.30% |
| 3 | 词 + 字符 TF-IDF 差值 + LR，C=0.3 | 69.68% | 字符 TF-IDF 差值 + LR，C=1 | 69.68% |
| 4 | 词 TF-IDF 差值 + LR，C=1 | 68.30% | 词 + 字符 TF-IDF 差值 + LR，C=3 | 68.94% |
| 5 | 字符 TF-IDF 差值 + LR，C=1 | 69.68% | 字符 TF-IDF 差值 + LR，C=3 | 67.98% |

### 当前框架的真实 API 运行记录

| 任务 | 后端／策略 | 运行状态 | 完成候选 | 成功调用 | 输入／输出 token | 费用 | 验证准确率 | 测试准确率 |
|---|---|---|---:|---:|---|---:|---:|---:|
| 优惠券 | 校内 API／多起点 | [未完成，ProxyError](runs/structured_comparison/coupon_campus/run_20261122_140821_0c1b1c/logs/run_summary.json) | 0 | 0 | 0／0 | ¥0.00 | — | — |

## 5. 结构化搜索扩展与验证记录

- 先训练基线并保存候选，再执行可选的校内计划和表格特征组合。校内失联时，保留已验证候选、运行状态和费用记录。
- 候选支持 accuracy、macro-F1、weighted-F1、ROC-AUC 和 LogLoss。OOF 融合用训练集内部折外概率选权重，再用原验证集算分。
- 概率按任务的类别顺序输出。测试集无标签时，最终阶段仍可保存预测和概率。

| 任务 | 验证配置 | 最佳单模型 | 双模型 OOF 融合 | API 调用 |
|---|---|---:|---:|---:|
| 优惠券 | [5 个常规候选 + 1 个融合](runs/hybrid_upgrade/coupon_oof/run_20261122_150056_898e52/logs/run_summary.json) | 76.70% | 76.55% | 0 |
| 标题 CTR | [5 个常规候选 + 1 个融合](runs/hybrid_upgrade/headlines_oof/run_20261122_150056_033f94/logs/run_summary.json) | 70.53% | 69.57% | 0 |

CPU 图像入口使用 scikit-learn Digits 做 10 类测试，验证 Macro-F1 为 96.39%。最终输出 360 条预测，使用外部保留的标签核对，准确率为 96.39%。

标题风格特征 + A/B 翻转增广和双向概率平均方案的验证准确率为 69.47%。

模拟 API 中断后，恢复运行沿用已有的两个候选，继续执行第三个候选。累计成功调用从 1 次增至 2 次。
