# Day 3 Template Reference Survey

## 1. 调研目的

本调研用于为 DeepLearning-Copilot 提取可复用、可计算的实验诊断模式。

项目只参考第三方项目公开的：

- 问题定义；
- 场景模式；
- 可观测信号；
- 所需输入字段；
- 官方默认参数或示例阈值；
- 评估思想。

项目不会：

- 复制第三方 Dataset；
- 复制第三方诊断模板；
- 复制第三方规则引擎实现；
- 让 LLM 决定 Ground Truth；
- 在当前阶段生成大规模 Dataset；
- 在当前阶段训练模型。

核心原则：

```text
Python deterministic Ground Truth
LLM must not assign labels
No third-party implementation code
```

Day 3 当前流程：

```text
Reference Survey
→ Scenario Taxonomy
→ Template Family Configuration
→ Configuration Validation
→ Small-scale Generation Test
```

---

## 2. 当前覆盖范围与待补任务

当前 Reference Survey 主要覆盖：

```text
experiment_diagnosis
```

包括：

- `overfitting`
- `underfitting`
- `optimization_problem`
- `class_imbalance`
- `data_quality_issue`
- `no_clear_issue`

Day 3 后续还必须分别补充：

```text
metric_interpretation
model_comparison
```

这两类任务也必须拥有独立的：

- `scenario_family`
- 输入字段定义
- 确定性 Ground Truth 逻辑
- `template_family`
- 配置验证规则
- Train / Validation / Test 覆盖

最终 Dataset 不能只有 `experiment_diagnosis`。

---

## 3. License 与引用政策

| Project | License | Usage policy |
| --- | --- | --- |
| Deepchecks | AGPL-3.0 | 只参考公开概念、定义、文档和参考阈值，不复制或改写实现代码。 |
| Cleanlab | Apache-2.0 | 优先参考公开概念与定义，项目自行实现所需逻辑。 |
| scikit-learn | BSD-3-Clause | 参考 learning curve、validation curve、overfitting 与 underfitting 的公开定义。 |
| imbalanced-learn | MIT | 参考类别不平衡指标和评估思想，项目自行实现特征计算。 |
| Keras | Apache-2.0 | 参考 callback 的公开语义和参数，不复制 callback 实现。 |

License 来源：

- Deepchecks：https://github.com/deepchecks/deepchecks/blob/main/LICENSE
- Cleanlab：https://github.com/cleanlab/cleanlab/blob/master/LICENSE
- scikit-learn：https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
- imbalanced-learn：https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/LICENSE
- Keras：https://github.com/keras-team/keras/blob/master/LICENSE

特别说明：

> Deepchecks 使用 AGPL-3.0。DeepLearning-Copilot 只参考其公开概念、定义、文档和参考阈值，must not copy Deepchecks implementation code。

---

## 4. 阈值解释政策

本调研中的数值分为三类：

- `reference_default`：第三方工具文档中的默认值；
- `reference_example`：第三方官方示例中的数值；
- `project_sampling_rule`：DeepLearning-Copilot 自己的数据采样规则。

第三方默认值和示例值不是通用科学定律。

DeepLearning-Copilot 必须：

- 将最终阈值保存在 `configs/threshold_bands_v1.json`；
- 排除 Standard Set 中的 borderline cases；
- 只生成明显正常或明显异常的标准样本；
- 使用 Python 计算特征；
- 使用 Python 规则生成 Ground Truth；
- 明确区分第三方参考值和项目采样规则。

---

# 5. Deepchecks 调研

官方来源：

- Train-Test Performance：  
  https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_train_test_performance.html
- Relative degradation condition：  
  https://docs.deepchecks.com/stable/api/generated/deepchecks.nlp.checks.model_evaluation.TrainTestPerformance.add_condition_train_test_relative_degradation_less_than.html
- Class Imbalance：  
  https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_class_imbalance.html
- Class-performance imbalance：  
  https://docs.deepchecks.com/dev/api/generated/deepchecks.tabular.checks.model_evaluation.TrainTestPerformance.add_condition_class_performance_imbalance_ratio_less_than.html
- Train-Test Samples Mix：  
  https://docs.deepchecks.com/stable/tabular/auto_checks/train_test_validation/plot_train_test_samples_mix.html
- Train-Test Validation Suite：  
  https://docs.deepchecks.com/stable/tabular/auto_tutorials/quickstarts/plot_quick_train_test_validation.html

License：AGPL-3.0

## 5.1 Train-Test Performance Degradation

1. Scenario Pattern

   训练集表现明显优于验证集或测试集。

2. Observable Signals

   - 相对 train-validation gap 较大；
   - 验证指标停滞或下降；
   - 部分类别的验证性能明显下降。

3. Required Inputs

   - `train_metric`
   - `validation_metric`
   - `metric_direction`
   - 可选 `validation_curve`
   - 可选逐类别指标

4. Reference Rule / Threshold

   - Deepchecks 文档中的默认相对退化条件为 `0.10`；
   - 官方 tabular 示例使用 `0.15`；
   - 这些是可配置条件，不是通用阈值。

5. 如何适配 DeepLearning-Copilot

   - 使用方向感知的 `relative_generalization_gap`；
   - 保留正常区、边界区和明显异常区；
   - Standard Set 中应同时要求 late validation degradation 等支持证据；
   - 在判断 overfitting 前先执行 data quality 和 optimization 的高优先级规则。

6. Source URL

   - https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_train_test_performance.html
   - https://docs.deepchecks.com/stable/api/generated/deepchecks.nlp.checks.model_evaluation.TrainTestPerformance.add_condition_train_test_relative_degradation_less_than.html

7. License

   AGPL-3.0，只参考概念和文档阈值。

## 5.2 Class Distribution Imbalance

1. Scenario Pattern

   最少类别的样本数明显少于最多类别。

2. Observable Signals

   - least-class / most-class ratio 很小；
   - minority class support 很低；
   - 标签分布明显偏斜。

3. Required Inputs

   - `class_counts`
   - 可选类别名称

4. Reference Rule / Threshold

   Deepchecks 官方示例使用 `0.15` 作为 least/most class ratio 条件。

5. 如何适配 DeepLearning-Copilot

   - 计算 `class_imbalance_ratio = min_count / max_count`；
   - 不允许仅凭 class count 判断最终问题；
   - Standard Set 应同时要求类别分布偏斜和少数类性能受损。

6. Source URL

   https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_class_imbalance.html

7. License

   AGPL-3.0，只参考概念。

## 5.3 Class-Performance Imbalance

1. Scenario Pattern

   最佳类别与最差类别之间存在明显性能差距。

2. Observable Signals

   - `class_performance_gap` 很大；
   - minority recall 或 minority F1 很低；
   - aggregate accuracy 掩盖少数类性能问题。

3. Required Inputs

   - `per_class_metric`
   - metric 名称
   - `metric_direction`
   - 可选 `class_counts`

4. Reference Rule / Threshold

   Deepchecks 文档中的可配置默认相对类别性能差异阈值为 `0.30`。

5. 如何适配 DeepLearning-Copilot

   - 保留现有 `class_performance_gap`；
   - 后续可以增加相对类别性能差异特征；
   - Standard Set 中应同时满足类别偏斜和性能受损；
   - 不直接把第三方相对差异阈值当成项目的绝对 gap 阈值。

6. Source URL

   https://docs.deepchecks.com/dev/api/generated/deepchecks.tabular.checks.model_evaluation.TrainTestPerformance.add_condition_class_performance_imbalance_ratio_less_than.html

7. License

   AGPL-3.0，只参考概念。

## 5.4 Train-Validation Sample Overlap

1. Scenario Pattern

   验证集或测试集样本同时出现在训练集中，导致评估结果过于乐观。

2. Observable Signals

   - 跨 split 的重复样本；
   - 非零 `split_overlap_rate`；
   - 异常偏高的验证表现。

3. Required Inputs

   - train sample identifiers 或 hashes
   - validation sample identifiers 或 hashes
   - `split_overlap_rate`

4. Reference Rule / Threshold

   - Deepchecks 使用可配置的 overlap condition；
   - 官方 suite 示例中常见最大条件为 `5%`；
   - 对干净 Standard Set，目标应为零重叠。

5. 如何适配 DeepLearning-Copilot

   - 保留 `split_overlap_rate`；
   - 普通 duplicate 和跨 split overlap 必须分开处理；
   - 项目的 Standard Set 使用更严格的确定性规则；
   - Template-level Split 还必须单独检查 `template_family_id`。

6. Source URL

   https://docs.deepchecks.com/stable/tabular/auto_checks/train_test_validation/plot_train_test_samples_mix.html

7. License

   AGPL-3.0，只参考概念。

## 5.5 Train-Validation Drift or Mismatch

1. Scenario Pattern

   训练集和验证集之间出现 feature drift、label drift、类别变化或预处理不一致。

2. Observable Signals

   - feature distribution drift；
   - label distribution drift；
   - 验证集出现新类别；
   - preprocessing 不一致；
   - domain classifier 可以明显区分两个 split。

3. Required Inputs

   - train / validation feature statistics
   - train / validation label statistics
   - drift score 或确定性 flag
   - 可选 category sets

4. Reference Rule / Threshold

   Deepchecks 官方 suite 示例包括：

   - feature drift 小于 `0.20`；
   - label drift 小于 `0.15`；
   - multivariate drift 小于 `0.25`。

   这些只是工具条件，不是通用阈值。

5. 如何适配 DeepLearning-Copilot

   - 第一版优先使用确定性 flag；
   - 保留 `distribution_shift_detected`；
   - 保留 `preprocessing_mismatch`；
   - 在项目实现确定性 drift calculator 前，不生成统计型 drift Ground Truth。

6. Source URL

   https://docs.deepchecks.com/stable/tabular/auto_tutorials/quickstarts/plot_quick_train_test_validation.html

7. License

   AGPL-3.0，只参考概念。

---

# 6. Cleanlab 调研

官方来源：

- Cleanlab 文档：  
  https://docs.cleanlab.ai/stable/
- Datalab issue types：  
  https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html
- Label-quality scores：  
  https://docs.cleanlab.ai/stable/cleanlab/rank.html
- Tabular label-issue tutorial：  
  https://docs.cleanlab.ai/stable/tutorials/clean_learning/tabular.html

License：Apache-2.0

## 6.1 Low Label Quality

1. Scenario Pattern

   当前标签获得的模型支持度较低，而其他标签获得更高支持度。

2. Observable Signals

   - label-quality score 较低；
   - given label 的预测概率较低；
   - 其他类别的预测概率明显更高；
   - 显式 label issue flag。

3. Required Inputs

   - `labels`
   - out-of-sample `pred_probs`
   - 可选 predicted labels
   - 可选 label-quality scores

4. Reference Rule / Threshold

   - Cleanlab label-quality score 范围为 `0` 到 `1`；
   - 分数越低，标签越可能有问题；
   - Cleanlab 没有提供适用于所有任务的统一 cutoff；
   - 推荐使用 cross-validation 得到的 out-of-sample probabilities。

5. 如何适配 DeepLearning-Copilot

   - 不使用任意固定 quality-score cutoff 生成 Ground Truth；
   - 优先生成确定性的 `label_noise_rate`；
   - label-quality score 只作为支持证据；
   - Standard Set 只包含明显的高噪声案例。

6. Source URL

   https://docs.cleanlab.ai/stable/cleanlab/rank.html

7. License

   Apache-2.0，只参考概念。

## 6.2 Dataset-Level Label Noise

1. Scenario Pattern

   数据集中有明显比例的样本可能被错误标注。

2. Observable Signals

   - label issue count 或 rate 较高；
   - 大量样本的 label-quality score 较低；
   - given label 与高置信度 alternative label 不一致。

3. Required Inputs

   - `label_noise_rate`
   - 可选 label issue count
   - total sample count
   - 可选 out-of-sample predicted probabilities

4. Reference Rule / Threshold

   Cleanlab 可以估计和排序 label issues，但没有提供适用于所有任务的统一噪声率阈值。

5. 如何适配 DeepLearning-Copilot

   - 使用项目自己的确定性 threshold bands；
   - 当前项目规则将 `label_noise_rate >= 0.20` 视为明显问题；
   - `label_noise_rate >= 0.35` 可对应 high severity；
   - 这些是 project sampling rules，不是 Cleanlab 阈值。

6. Source URL

   https://docs.cleanlab.ai/stable/tutorials/clean_learning/tabular.html

7. License

   Apache-2.0，只参考概念。

## 6.3 Near-Duplicate or Duplicate Samples

1. Scenario Pattern

   样本完全重复，或者在 feature / embedding 空间中异常接近。

2. Observable Signals

   - duplicate hash match；
   - nearest-neighbour distance 很小；
   - 出现近似重复样本簇；
   - train 和 validation 之间存在重复。

3. Required Inputs

   - sample hashes 或标准化记录
   - feature embeddings 或 KNN graph
   - `duplicate_rate`
   - 可选 split membership

4. Reference Rule / Threshold

   Cleanlab 使用最近邻距离与数据集整体距离分布进行比较，没有适用于所有数据集的统一 cutoff。

5. 如何适配 DeepLearning-Copilot

   - 第一版使用明确的 `duplicate_rate`；
   - 跨 split 重叠使用 exact hash；
   - 暂缓 embedding-based near-duplicate 场景；
   - 普通重复和 split leakage 必须分别建模。

6. Source URL

   https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html

7. License

   Apache-2.0，只参考概念。

## 6.4 Non-IID or Distribution Shift

1. Scenario Pattern

   数据顺序或数据分布违反 IID 假设，或者不同 split 来自不同分布。

2. Observable Signals

   - non-IID test 显著；
   - 出现 drift 或 changepoint；
   - 相邻样本异常相似；
   - 明确的 train-validation domain shift。

3. Required Inputs

   - feature embeddings 或 KNN graph
   - sample order
   - 可选 p-value
   - 可选确定性 shift flag

4. Reference Rule / Threshold

   Cleanlab 文档使用 `p < 0.05` 表示统计上显著的 non-IID 证据。

5. 如何适配 DeepLearning-Copilot

   - 第一版使用 `distribution_shift_detected`；
   - 未实现统计检验前不得伪造 p-value；
   - statistical non-IID 场景暂缓进入 Standard Set。

6. Source URL

   https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html

7. License

   Apache-2.0，只参考概念。

---

# 7. scikit-learn 调研

官方来源：

- Validation and learning curves：  
  https://scikit-learn.org/stable/modules/learning_curve.html
- Underfitting versus overfitting：  
  https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
- Learning-curve API：  
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html

License：BSD-3-Clause

## 7.1 High Training Score and Low Validation Score

1. Scenario Pattern

   训练表现很好，但验证表现明显较差。

2. Observable Signals

   - high training score；
   - low validation score；
   - 持续或逐渐扩大的 generalization gap。

3. Required Inputs

   - train score
   - validation score
   - metric direction
   - 可选 epoch curve 或 training-size curve

4. Reference Rule / Threshold

   scikit-learn 给出定性模式，但不提供通用数值阈值。

5. 如何适配 DeepLearning-Copilot

   - 映射到 `overfitting`；
   - 使用项目自己的 relative gap bands；
   - Standard Set 使用明显分离的数值；
   - 有曲线时优先加入 late degradation 或 gap growth 证据。

6. Source URL

   https://scikit-learn.org/stable/modules/learning_curve.html

7. License

   BSD-3-Clause，只参考概念。

## 7.2 Low Training and Validation Scores

1. Scenario Pattern

   训练与验证表现都收敛到相近但较差的水平。

2. Observable Signals

   - weak training score；
   - weak validation score；
   - train-validation gap 较小；
   - 增加样本或 epoch 后改善有限。

3. Required Inputs

   - train score
   - validation score
   - explicit reference 或 baseline
   - 可选 training-size curve
   - 可选 training plateau

4. Reference Rule / Threshold

   scikit-learn 没有给出跨任务通用的“低性能”阈值。

5. 如何适配 DeepLearning-Copilot

   - 映射到 `underfitting`；
   - 必须提供 `reference_performance`；
   - 禁止使用 `accuracy < 0.70` 之类的通用规则；
   - 必须区分稳定 underfitting 和不稳定 optimization。

6. Source URL

   https://scikit-learn.org/stable/modules/learning_curve.html

7. License

   BSD-3-Clause，只参考概念。

## 7.3 Validation-Curve Bias and Variance Pattern

1. Scenario Pattern

   模型在不同超参数区域分别表现出 underfitting 和 overfitting。

2. Observable Signals

   - 多个参数值对应的 train / validation curves；
   - high-bias 区域中两种 score 都较低；
   - high-variance 区域中 score gap 很大；
   - 中间区域可能获得最佳 validation score。

3. Required Inputs

   - hyperparameter name
   - ordered parameter values
   - train scores
   - validation scores
   - metric direction

4. Reference Rule / Threshold

   没有通用阈值，必须根据多个参数值之间的相对关系判断。

5. 如何适配 DeepLearning-Copilot

   - 后续用于 `metric_interpretation` 和 `model_comparison`；
   - 不强行放入 single-run experiment diagnosis；
   - 在实现确定性 multi-run Ground Truth 后再加入。

6. Source URL

   https://scikit-learn.org/stable/modules/learning_curve.html

7. License

   BSD-3-Clause，只参考概念。

---

# 8. imbalanced-learn 调研

官方来源：

- Metrics：  
  https://imbalanced-learn.org/stable/references/metrics.html
- Imbalanced classification report：  
  https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.classification_report_imbalanced.html
- Geometric mean：  
  https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.geometric_mean_score.html

License：MIT

## 8.1 Minority-Class Performance Collapse

1. Scenario Pattern

   整体指标看起来正常，但一个或多个少数类的 recall、specificity、F1 或 G-mean component 很差。

2. Observable Signals

   - low minority recall；
   - low minority F1；
   - large per-class performance gap；
   - 某一类别几乎无法被识别。

3. Required Inputs

   - `class_counts`
   - per-class precision / recall / F1
   - 可选 specificity
   - 可选 confusion matrix

4. Reference Rule / Threshold

   imbalanced-learn 定义了相关指标，但没有规定统一失败阈值。

5. 如何适配 DeepLearning-Copilot

   - 保留通用 `per_class_metric`；
   - 后续增加明确的 per-class recall 和 F1 字段；
   - Standard Set 必须同时出现 distribution skew 和 performance damage。

6. Source URL

   https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.classification_report_imbalanced.html

7. License

   MIT，只参考概念。

## 8.2 Aggregate Metric Masks Imbalance

1. Scenario Pattern

   Accuracy 或 weighted metric 较高，但 macro 或 balanced metric 明显较低。

2. Observable Signals

   - large accuracy–macro-F1 gap；
   - low balanced accuracy；
   - low G-mean；
   - 明显的 per-class recall 差异。

3. Required Inputs

   - accuracy
   - macro-F1
   - 可选 balanced accuracy
   - 可选 G-mean
   - class counts

4. Reference Rule / Threshold

   - G-mean 最佳值为 `1`，最差值为 `0`；
   - G-mean 为零可能表示至少一个类别未被识别；
   - imbalanced-learn 没有提供统一异常 cutoff。

5. 如何适配 DeepLearning-Copilot

   - 保留 `accuracy_macro_f1_gap`；
   - 后续可增加 `geometric_mean_score`；
   - 使用项目自己的明显异常采样区间；
   - 禁止只根据 accuracy 判断 class imbalance。

6. Source URL

   https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.geometric_mean_score.html

7. License

   MIT，只参考概念。

---

# 9. Keras 调研

官方来源：

- EarlyStopping：  
  https://keras.io/api/callbacks/early_stopping/
- ReduceLROnPlateau：  
  https://keras.io/api/callbacks/reduce_lr_on_plateau/
- TerminateOnNaN：  
  https://keras.io/api/callbacks/terminate_on_nan/

License：Apache-2.0

## 9.1 Training Plateau

1. Scenario Pattern

   被监控指标在连续多个 epoch 中没有明显改善。

2. Observable Signals

   - 连续多个 epoch 无改善；
   - improvement 小于 `min_delta`；
   - patience 被耗尽；
   - 可选 early-stopping event。

3. Required Inputs

   - metric curve
   - metric direction
   - `min_delta`
   - `patience`
   - 可选 monitoring start epoch

4. Reference Rule / Threshold

   Keras `EarlyStopping` 默认参数包括：

   - `monitor="val_loss"`
   - `min_delta=0`
   - `patience=0`
   - `start_from_epoch=0`
   - `restore_best_weights=False`

   这些是 callback 默认值，不是 underfitting 或 optimization failure 的科学定义。

5. 如何适配 DeepLearning-Copilot

   - 保留 `plateau_streak`；
   - 对不同量纲的指标使用相对改善；
   - plateau 只能作为支持证据；
   - plateau 与 reference shortfall 组合后可支持 underfitting；
   - 当前 rule engine 不应把 plateau 单独判定为 optimization problem。

6. Source URL

   https://keras.io/api/callbacks/early_stopping/

7. License

   Apache-2.0，只参考概念。

## 9.2 Plateau Followed by Learning-Rate Reduction

1. Scenario Pattern

   监控指标停止改善，经过 patience window 后 learning rate 被降低。

2. Observable Signals

   - validation metric plateau；
   - learning-rate reduction event；
   - 多次降低学习率后仍没有恢复；
   - learning rate 到达最小值。

3. Required Inputs

   - monitored metric curve
   - learning-rate curve
   - reduction event epochs
   - `factor`
   - `patience`
   - `min_delta`
   - `min_lr`

4. Reference Rule / Threshold

   Keras `ReduceLROnPlateau` 默认参数包括：

   - `factor=0.1`
   - `patience=10`
   - `min_delta=0.0001`
   - `cooldown=0`
   - `min_lr=0.0`

5. 如何适配 DeepLearning-Copilot

   - 这些数值只能视为 callback defaults；
   - feature calculator 支持 learning-rate history 后才能加入；
   - 多次降低学习率但没有恢复，可作为未来 optimization scenario；
   - 第一版 Standard Set 暂缓此场景。

6. Source URL

   https://keras.io/api/callbacks/reduce_lr_on_plateau/

7. License

   Apache-2.0，只参考概念。

## 9.3 NaN or Inf Loss

1. Scenario Pattern

   训练过程中产生 NaN 或 Inf loss，导致训练无法正常继续。

2. Observable Signals

   - NaN loss；
   - Inf loss；
   - numerical instability event；
   - training early termination。

3. Required Inputs

   - `nan_or_inf` flag
   - 可选 loss curve
   - 可选 failure epoch

4. Reference Rule / Threshold

   这是布尔型异常信号。只要出现 NaN 或 Inf loss，就说明存在明确的 optimization abnormality。

5. 如何适配 DeepLearning-Copilot

   - 直接映射到 `optimization_problem`；
   - 保留现有 `nan_or_inf` flag；
   - 根据当前规则设为 high severity；
   - Standard Set 中不得同时加入 data-quality trigger。

6. Source URL

   https://keras.io/api/callbacks/terminate_on_nan/

7. License

   Apache-2.0，只参考概念。

---

# 10. Proposed Scenario Taxonomy

本表是调研结果，还不是 `configs/template_families_v1.json`。

| Issue | Proposed scenario_family | Current compatibility | Main source |
| --- | --- | --- | --- |
| overfitting | `generalization_gap_with_late_degradation` | Ready | Deepchecks + scikit-learn |
| overfitting | `validation_loss_best_then_rises` | Ready | scikit-learn |
| underfitting | `low_scores_vs_explicit_reference` | Ready | scikit-learn |
| underfitting | `reference_shortfall_with_plateau` | Ready | scikit-learn + Keras |
| optimization_problem | `strong_loss_oscillation` | Ready | Existing project feature logic |
| optimization_problem | `nan_or_inf_training` | Ready | Keras |
| optimization_problem | `repeated_lr_reduction_without_recovery` | Deferred：需要 learning-rate history | Keras |
| class_imbalance | `distribution_skew_with_classwise_collapse` | Ready | Deepchecks + imbalanced-learn |
| class_imbalance | `accuracy_masks_low_macro_f1` | Ready | imbalanced-learn |
| class_imbalance | `minority_recall_collapse` | Ready：暂时使用通用 per-class metric | imbalanced-learn |
| data_quality_issue | `high_label_noise_rate` | Ready | Cleanlab |
| data_quality_issue | `high_duplicate_rate` | Ready | Cleanlab |
| data_quality_issue | `train_validation_sample_overlap` | Ready | Deepchecks |
| data_quality_issue | `preprocessing_mismatch` | Ready | Deepchecks-inspired |
| data_quality_issue | `explicit_distribution_shift` | Ready：使用确定性 flag | Deepchecks + Cleanlab |
| data_quality_issue | `statistical_non_iid_detection` | Deferred：需要确定性统计检验 | Cleanlab |
| no_clear_issue | `stable_convergence_small_gap` | Ready | Negative control |
| no_clear_issue | `balanced_classwise_performance` | Ready | Negative control |

---

# 11. Template-Family 层级

后续配置必须使用三层结构：

```text
issue
→ scenario_family
→ template_family
```

示例：

```text
overfitting
→ late_validation_degradation
→ epoch_table_report
→ narrative_training_summary
→ experiment_tracking_note
```

`scenario_family` 表示稳定的诊断机制。

`template_family` 表示表达该场景的稳定输入结构和呈现结构。

以下变化不能形成新的 `template_family`：

- 只修改数值；
- 只修改类别名称；
- 只修改 Dataset 名称；
- 只替换同义词；
- 保持推理结构不变，只修改 metric 名称；
- 只调整句子顺序；
- 只做表面 paraphrase。

新的 `template_family` 必须改变输入或呈现结构，例如：

- epoch table；
- compact metric dictionary；
- experiment-tracking report；
- natural-language training log；
- model-card excerpt；
- debugging ticket；
- class-performance report；
- comparison table。

---

# 12. Standard Split Policy

Standard split unit = `template_family_id`.

标准划分规则：

- 每个 `template_family_id` 只能属于 Train、Validation、Test 中的一个 split；
- numeric variants 必须继承所属 `template_family_id` 的 split；
- paraphrase variants 必须继承所属 `template_family_id` 的 split；
- 同一个 `template_family_id` 禁止出现在多个 Standard Split；
- scenario_family can span multiple Standard Splits through different template_family_id values；
- `scenario_family` 可以通过不同的 `template_family_id` 分布在 Train、Validation、Test；
- 不要求整个 `scenario_family` 只能属于一个 Standard Split；
- completely unseen `scenario_family` 留给 Hard Test；
- split assignment 必须发生在生成样本之前；
- 禁止生成 Dataset 后再执行 random split；
- 禁止把同一 template family 的单条样本随机分配到不同 split。

示例：

```text
scenario_family:
late_validation_degradation

Train template_family_id:
OF_LATE_DEGRADATION_EPOCH_TABLE

Validation template_family_id:
OF_LATE_DEGRADATION_TRACKING_REPORT

Test template_family_id:
OF_LATE_DEGRADATION_NARRATIVE_LOG
```

上面三个 `template_family_id` 属于同一个 `scenario_family`，但它们的输入呈现结构不同。

只修改数字或措辞不能生成新的 template family，也不能以此为理由进入另一个 split。

---

# 13. Dataset 政策

## 13.1 Standard Set

Standard Set 必须遵守：

- 每个样本只有一个 primary issue；
- 排除 borderline cases；
- 使用 Python deterministic Ground Truth；
- LLM must not assign labels；
- numeric 和 paraphrase variants 禁止跨越 family split；
- 不复制第三方文档内容作为训练样本；
- 所有样本必须通过现有 Python diagnosis pipeline；
- task 和 label 必须符合现有 taxonomy 与 output schema。

## 13.2 Hard Test

Hard Test 暂缓实现，以后可以包含：

- borderline threshold cases；
- multi-issue cases；
- conflicting evidence；
- completely unseen scenario_family；
- unseen scenario combinations；
- 不完整输入；
- noisy descriptions；
- distractor metrics；
- ambiguous evidence。

---

# 14. 与 Day 2 组件的兼容性

标记为 `Ready` 的 scenario family 必须兼容：

```text
configs/taxonomy_v1.json
configs/diagnosis_rules_v1.json
configs/feature_definitions_v1.json
configs/threshold_bands_v1.json
configs/output_schema_v1.json
configs/output_vocabulary_v1.json
configs/recommendation_mapping_v1.json
```

Ground Truth 必须继续经过：

```text
src/evaluation/feature_calculator.py
src/evaluation/rule_engine.py
src/evaluation/diagnosis_pipeline.py
src/evaluation/output_validator.py
src/evaluation/ground_truth_builder.py
```

标记为 `Deferred` 的 scenario family，在缺少确定性输入、特征计算或规则时，不得进入 Standard Set。

---

# 15. 结论

第一版 Template Family Configuration 只使用标记为 `Ready` 的 scenario family。

第三方默认值可以帮助设计 sampling bands，但项目最终阈值仍由以下组件控制：

```text
configs/threshold_bands_v1.json
src/evaluation/feature_calculator.py
src/evaluation/rule_engine.py
```

当前 Reference Survey 主要覆盖：

```text
experiment_diagnosis
```

Day 3 后续必须继续补充：

```text
metric_interpretation
model_comparison
```

最终 Dataset 不能只包含 `experiment_diagnosis`。

No third-party implementation code：本调研没有复制任何第三方实现代码或 Dataset。
