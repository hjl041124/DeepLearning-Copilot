# DeepLearning-Copilot Rule Design References
# （DeepLearning-Copilot 規則設計參考）

## Purpose（目的）

This document records the external references used to design
the deterministic Python-based diagnosis rules.

本文件記錄 DeepLearning-Copilot 的 Python deterministic rules
（Python 確定性規則）所參考的成熟工具、官方文檔和開源實現。

Important:
The project does NOT directly copy third-party rule-engine code.
References are used to inform concepts, signals, and reasonable
starting thresholds, which are then adapted to this project's
deep-learning experiment diagnosis task.

重要原則：
1. 不直接複製第三方 Rule Engine（規則引擎）程式碼。
2. Ground Truth（標準答案）最終由本專案自己的 Python 程式生成。
3. 第三方 threshold（閾值）只作 Reference Anchor（參考錨點）。
4. 所有規則必須適配 DeepLearning-Copilot 的任務。
5. Test Set（測試集）建立後不得根據模型結果反向修改規則。


---

## 1. Overfitting / Underfitting
## （過擬合 / 欠擬合）

### Reference（參考）

scikit-learn
Validation curves / Learning curves

Official documentation:
https://scikit-learn.org/stable/modules/learning_curve.html

License:
BSD-3-Clause

### Reference concept（參考概念）

Underfitting（欠擬合）:
- Training score（訓練分數）低
- Validation score（驗證分數）低

Overfitting（過擬合）:
- Training score（訓練分數）高
- Validation score（驗證分數）明顯較低

Healthy / no clear issue（正常 / 無明顯問題）:
- Training and validation performance are both reasonably strong
- Gap is limited

### Adaptation for DeepLearning-Copilot
### （本專案適配）

Do NOT define underfitting using a universal rule such as:
accuracy < 0.70.

Instead use relative relationships between:
- training performance
- validation performance
- generalization gap
- training stability
- metric trend across epochs


---

## 2. Train-Validation Relative Degradation
## （訓練-驗證相對性能退化）

### Reference（參考）

Deepchecks
TrainTestPerformance

Official documentation:
https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_train_test_performance.html

License:
AGPL-3.0+

### Reference threshold（參考閾值）

Deepchecks example condition:

train-test relative degradation < 0.15

Reference anchor:
0.15 = 15%

### Adaptation for DeepLearning-Copilot
### （本專案適配）

The value 0.15 is NOT treated as universal ground truth.

It is used as a reference anchor for defining:
- relative_generalization_gap
- moderate / strong generalization degradation

Additional evidence should include:
- train metric trend
- validation metric trend
- validation degradation or stagnation
- optimization stability


---

## 3. Class Imbalance
## （類別不平衡）

### Reference（參考）

Deepchecks
ClassImbalance

Source:
https://github.com/deepchecks/deepchecks/blob/main/deepchecks/tabular/checks/data_integrity/class_imbalance.py

License:
AGPL-3.0+

### Reference definition（參考定義）

class_imbalance_ratio:

least_frequent_class_count
/
most_frequent_class_count

Deepchecks default reference threshold:
0.10

### Adaptation for DeepLearning-Copilot
### （本專案適配）

class_ratio <= 0.10 should NOT automatically imply that
the model is suffering from class imbalance.

The diagnosis should additionally consider one or more of:
- minority recall
- minority F1
- macro-F1
- overall accuracy
- per-class metric gap

This prevents class distribution skew from being confused with
actual model-performance degradation.


---

## 4. Optimization Plateau
## （優化停滯）

### Reference（參考）

Keras
EarlyStopping

Official documentation:
https://keras.io/api/callbacks/early_stopping/

License:
Apache-2.0

### Reference concept（參考概念）

Plateau detection can be expressed with:
- monitor
- min_delta
- patience
- start_from_epoch

An improvement smaller than min_delta can be treated as
no meaningful improvement.

Several consecutive epochs without meaningful improvement
can indicate a plateau.

### Adaptation for DeepLearning-Copilot
### （本專案適配）

Use this concept to define:
- training_loss_plateau
- validation_loss_plateau

Do NOT directly label every plateau as optimization_problem.

A plateau with poor train performance may support:
underfitting / insufficient learning.

A plateau caused by unstable learning behavior may support:
optimization_problem.


---

## 5. Label Quality
## （標籤品質）

### Reference（參考）

Cleanlab
Label issue detection

Official documentation:
https://docs.cleanlab.ai/

License:
Apache-2.0

### Reference concept（參考概念）

Label-quality estimation should use:
out-of-sample predictions

Predictions on the same samples used for model fitting can be
overfit and should not be treated as reliable evidence of
label errors.

### Adaptation for DeepLearning-Copilot
### （本專案適配）

The first dataset version will not run Cleanlab itself.

Instead, synthetic experiment records may explicitly provide:
- label_noise_rate
- conflicting_label_rate
- suspected_bad_labels

These values are treated as externally measured dataset statistics.

The LLM must NOT invent them.


---

## 6. Imbalanced Classification Metrics
## （不平衡分類指標）

### Reference（參考）

imbalanced-learn

Official repository:
https://github.com/scikit-learn-contrib/imbalanced-learn

License:
MIT

### Reference concept（參考概念）

Aggregate accuracy alone may hide poor minority-class performance.

Useful signals include:
- per-class recall
- per-class F1
- macro-F1
- class distribution

### Adaptation for DeepLearning-Copilot
### （本專案適配）

Class imbalance cases should preferably include both:
1. distribution evidence
2. class-wise performance evidence


---

# Licensing Policy（許可證策略）

## scikit-learn
License:
BSD-3-Clause

Usage:
Concept/reference only.

## imbalanced-learn
License:
MIT

Usage:
Concept/reference only.

## Keras
License:
Apache-2.0

Usage:
Concept/reference only.

## Cleanlab
License:
Apache-2.0

Usage:
Concept/reference only.

## Deepchecks
License:
AGPL-3.0+

Usage:
Reference concepts and documented thresholds only.

DeepLearning-Copilot will NOT directly copy Deepchecks source code.
The project's rule engine will be independently implemented.


---

# Design Decision（設計決策）

DeepLearning-Copilot will prefer:

relative rules
（相對規則）

over

universal absolute task-performance thresholds
（通用絕對性能閾值）

Examples:

Preferred:
relative_generalization_gap

Avoid:
accuracy < 0.70 means underfitting

Preferred:
minority / majority class ratio + class-wise performance gap

Avoid:
imbalanced class counts automatically mean class_imbalance

Preferred:
trend + plateau + stability

Avoid:
one bad epoch automatically means optimization_problem


---

# Status（狀態）

Reference review:
COMPLETED

Source-code reuse:
NONE

Next:
Define project-specific quantitative features and threshold bands.
