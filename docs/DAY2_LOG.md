# Day 2 Experiment Log（第二天實驗記錄）

## Goal（目標）

完成 DeepLearning-Copilot 的：

- Task Taxonomy（任務分類體系）
- Diagnosis Rules（診斷規則）
- Quantitative Features（定量特徵）
- Threshold Bands（閾值區間）
- Feature Calculator（特徵計算器）
- Rule Engine（規則引擎）
- Diagnosis Pipeline（診斷流程）
- JSON Schema（JSON 結構規範）
- Ground Truth Builder（標準答案構建器）
- Configuration Consistency Check（配置一致性檢查）

## Task Types（任務類型）

1. experiment_diagnosis（實驗診斷）
2. metric_interpretation（指標解讀）
3. model_comparison（模型比較）

## Primary Issues（主要問題）

1. overfitting（過擬合）
2. underfitting（欠擬合 / 學習不足）
3. optimization_problem（優化問題）
4. class_imbalance（類別不平衡）
5. data_quality_issue（資料品質問題）
6. no_clear_issue（無明顯問題）
7. not_applicable（不適用）

## Standard Set Policy（標準集策略）

- Single primary issue per sample（每個樣本只含一個主要問題）
- Multi-issue cases reserved for Hard Test（多問題案例留給困難測試集）
- Borderline cases excluded from Standard Set（邊界案例暫不進標準集）
- Ground Truth is generated deterministically by Python（標準答案由 Python 確定性生成）
- LLM does not assign Ground Truth（LLM 不決定標準答案）

## Rule Design References（規則設計參考）

Concepts were informed by:

- scikit-learn
- Deepchecks
- Keras
- Cleanlab
- imbalanced-learn

No third-party rule-engine source code was copied.

## Core Pipeline（核心流程）

Raw Experiment Data
→ Feature Calculator
→ Rule Engine
→ Diagnosis
→ Recommendation Mapping
→ Ground Truth Builder
→ JSON Schema Validation

## Tests Passed（已通過測試）

- FEATURE CALCULATOR TESTS PASSED
- RULE ENGINE TESTS PASSED
- END-TO-END DIAGNOSIS TESTS PASSED
- OUTPUT SCHEMA VALIDATION TESTS PASSED
- GROUND TRUTH BUILDER TESTS PASSED
- DAY2 CONFIG CONSISTENCY CHECK PASSED

## Consistency Check（配置一致性）

- Task types: 3
- Primary issues: 7
- Feature definitions: 17
- Evidence codes: 20
- Action codes: 24

## Important Notes（重要說明）

- Thresholds are dataset-generation rules, not universal scientific thresholds.
- Borderline cases will be handled separately in Hard Test.
- No large-scale dataset has been generated yet.
- No model training has been performed.
- No experimental metric has been fabricated.

## Day 2 Status（第二天狀態）

COMPLETED
