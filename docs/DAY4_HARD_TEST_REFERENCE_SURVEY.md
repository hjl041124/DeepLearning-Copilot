# Day 4 Hard Test Reference Survey

## 1. 目的

Day 4 用于设计 DeepLearning-Copilot 的 Hard Test（困难测试集）。

Hard Test 不用于：

- Training（训练）
- Validation（验证）
- QLoRA（量化低秩适配微调）

Hard Test 只用于：

- Base Model Evaluation（基线模型评估）
- Fine-tuned Model Evaluation（微调模型评估）
- Behavioral Robustness Evaluation（行为鲁棒性评估）

核心原则：

```text
Evaluation only
Python deterministic Ground Truth
LLM must not assign labels
No third-party implementation code
```

---

## 2. CheckList 参考

Reference:

- Paper:
  https://aclanthology.org/2020.acl-main.442/
- GitHub:
  https://github.com/marcotcr/checklist

License:

```text
MIT
```

CheckList 提供三种重要 Behavioral Test（行为测试）：

### MFT — Minimum Functionality Test（最小功能测试）

用于检查模型是否具备某个能力的基本功能。

DeepLearning-Copilot 的 Standard Set（标准数据集）已经承担大部分 MFT-like（类似最小功能测试）作用，因此 Day 4 不重复建立大量 MFT。

### INV — Invariance Test（不变性测试）

对输入进行 Label-preserving Perturbation（标签保持扰动）。

要求：

```text
Input
↓
加入无关扰动
↓
Expected Decision 不变
```

DeepLearning-Copilot 对应：

```text
Invariance Distractor Test
```

例如加入：

- GPU
- seed
- batch_size
- run_name
- optimizer metadata

只要这些字段不参与现有 Ground Truth Rule（标准答案规则），诊断结果就不得变化。

### DIR — Directional Expectation Test（方向性预期测试）

输入中的关键因素发生有意义变化后，模型输出应按照已知方向变化。

DeepLearning-Copilot 对应：

```text
Directional Boundary Test
```

例如：

```text
relative_generalization_gap
threshold - epsilon
↓
threshold + epsilon
```

在其他控制变量不变时，Ground Truth 应发生预期变化。

---

## 3. Hypothesis 参考

Reference:

- Documentation:
  https://hypothesis.readthedocs.io/en/latest/
- GitHub:
  https://github.com/HypothesisWorks/hypothesis

License:

```text
MPL-2.0
```

Hypothesis 是 Property-based Testing（基于属性的测试）工具。

DeepLearning-Copilot 后续只使用其测试思想或测试库来检查：

- Boundary Sampling（边界采样）
- Edge Case（边界案例）
- Generator Invariant（生成器不变量）
- Threshold Crossing（阈值跨越）
- Pair Property（样本对属性）

Hypothesis 不负责：

- Dataset Ground Truth
- Diagnosis Label
- LLM Labeling

Ground Truth 继续由项目自己的 Python Rule Engine（Python 规则引擎）决定。

---

## 4. Deepchecks 参考

Reference:

- Documentation:
  https://docs.deepchecks.com/stable/
- GitHub:
  https://github.com/deepchecks/deepchecks

License:

```text
AGPL-3.0
```

只参考：

- Data Integrity（数据完整性）
- Train-Test Validation（训练测试验证）
- Configurable Condition（可配置条件）
- Suite（检查套件）

不复制 Deepchecks 实现代码。

Day 4 数据质量相关 Hard Test 可以使用项目当前已有的：

- label_noise_rate
- split_overlap_rate
- preprocessing_mismatch
- distribution_shift_detected

但阈值仍由：

```text
configs/threshold_bands_v1.json
```

控制。

---

## 5. HELM 参考

Reference:

https://github.com/stanford-crfm/helm

License:

```text
Apache-2.0
```

采用的主要思想：

- standardized evaluation（标准化评估）
- reproducibility（可复现性）
- multi-metric evaluation（多指标评估）
- scenario-level reporting（场景级报告）

DeepLearning-Copilot 的 Hard Test 不只报告总体 Accuracy。

必须同时报告不同 Test Slice（测试切片）。

---

## 6. Robustness Gym 参考

Reference:

https://github.com/robustness-gym/robustness-gym

License:

```text
Apache-2.0
```

采用的主要思想：

- Slice-based Evaluation（切片评估）
- Subpopulation Analysis（子群体分析）
- Failure Analysis（失败分析）

DeepLearning-Copilot 后续至少分别报告：

```text
Directional Boundary Accuracy
Invariance Success Rate
Invariance Violation Rate
Priority Composition Accuracy
Hard Test Family Accuracy
Overall Hard Test Accuracy
JSON Schema Valid Rate
```

---

## 7. Hard Test Property 设计

Day 4 v1 使用三种 Test Property（测试属性）。

### 7.1 Directional Boundary Test（方向性边界测试）

结构：

```text
pair_id

Case A:
target_feature = threshold - epsilon

Case B:
target_feature = threshold + epsilon
```

要求：

- Pair（样本对）的非关键变量尽量保持一致；
- 只有目标 Feature（特征）跨越 Threshold（阈值）；
- Ground Truth 必须通过 Python Rule Engine 重新计算；
- 不硬编码最终 Label；
- 记录 Expected Direction（预期方向）。

第一版覆盖：

- relative_generalization_gap
- class_imbalance_ratio
- label_noise_rate

### 7.2 Invariance Distractor Test（不变性干扰测试）

结构：

```text
pair_id

Base Sample
↓
加入与 Ground Truth 无关的 Metadata
↓
Perturbed Sample
```

要求：

```text
decision_signature(base)
==
decision_signature(perturbed)
```

decision_signature 暂定为：

```text
task_type
primary_issue
severity
evidence_codes
recommended_action_codes
```

Explanation（解释文本）不用于 Invariance Equality（不变性相等判断）。

第一版覆盖：

- overfitting + irrelevant metadata
- data_quality_issue + irrelevant metadata
- model_comparison + irrelevant metadata

### 7.3 Priority Composition Test（优先级组合测试）

同一 Raw Record（原始记录）同时触发多个已知问题。

Primary Issue 必须遵守：

```text
data_quality_issue
>
optimization_problem
>
class_imbalance
>
overfitting
>
underfitting
>
no_clear_issue
```

第一版覆盖：

- data_quality_issue + overfitting
- optimization_problem + overfitting
- class_imbalance + overfitting
- data_quality_issue + optimization_problem
- data_quality_issue + class_imbalance
- optimization_problem + underfitting

这些组合不进入 Standard Set。

---

## 8. Hard Test 与 Standard Set 的区别

Standard Set：

```text
Single Primary Issue
Clear Cases
Template-level Generalization
No borderline cases
```

Hard Test：

```text
Threshold Boundary
Paired Counterfactual Input
Invariance Perturbation
Multiple Simultaneous Triggers
Rule Priority
Unseen Evidence Composition
```

Hard Test 不参与训练。

---

## 9. Hard Test v1 规模

```text
Directional Boundary:
3 families × 20 samples = 60

Invariance Distractor:
3 families × 20 samples = 60

Priority Composition:
6 families × 20 samples = 120

Total:
240 samples
```

对于 Pair-based Test（基于样本对的测试）：

```text
20 samples = 10 pairs × 2 samples
```

---

## 10. 结论

Day 4 Hard Test 不再简单定义为“更难的数据”。

每一个 Hard Test Family 必须具有明确的：

```text
test_property
controlled_variables
changed_variables
expected_relation
Ground Truth recomputation rule
evaluation metric
```

所有 Ground Truth 必须继续由 DeepLearning-Copilot 自己的 Python Rule Engine 产生。

No third-party implementation code is copied.
