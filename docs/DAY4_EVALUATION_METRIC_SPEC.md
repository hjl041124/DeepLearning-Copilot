# Day 4 Evaluation Metric Specification

## 1. 目的

本文件固定 DeepLearning-Copilot 后续所有 Base Model（基线模型）和 Fine-tuned Model（微调模型）的评估规则。

后续模型比较必须：

```text
Same Dataset
Same Parser
Same Output Schema
Same Failure Policy
Same Metric Implementation

cd <PROJECT_ROOT>

mkdir -p configs docs scripts


# ============================================================
# 1. Evaluation Metric Specification（评估指标规范）
# ============================================================

cat > configs/evaluation_metrics_v1.json <<'EOF'
{
  "version": "1.0",
  "description": "Evaluation metric specification for DeepLearning-Copilot Standard Test and Hard Test.",
  "evaluation_policy": {
    "same_parser_for_all_models": true,
    "same_schema_validator_for_all_models": true,
    "same_failure_policy_for_all_models": true,
    "invalid_or_unparseable_output_is_not_repaired_for_main_evaluation": true,
    "explanation_exact_string_match": false,
    "evidence_codes_are_unordered_sets": true,
    "recommended_action_codes_are_unordered_sets": true,
    "main_comparison_requires_same_test_set": true
  },
  "references": [
    {
      "name": "scikit-learn",
      "usage": [
        "accuracy_definition",
        "macro_f1_definition",
        "multiclass_f1_reporting"
      ]
    },
    {
      "name": "CheckList",
      "usage": [
        "directional_pair_evaluation",
        "invariance_relation_evaluation"
      ]
    },
    {
      "name": "HELM",
      "usage": [
        "scenario_level_metric_reporting",
        "standardized_evaluation"
      ]
    }
  ],
  "core_prediction_fields": [
    "task_type",
    "primary_issue",
    "severity",
    "evidence_codes",
    "recommended_action_codes"
  ],
  "standard_test_metrics": [
    {
      "metric_id": "parse_success_rate",
      "level": "all_samples",
      "definition": "Fraction of model responses that can be parsed into a JSON object."
    },
    {
      "metric_id": "json_schema_valid_rate",
      "level": "all_samples",
      "definition": "Fraction of all model responses that parse successfully and pass the project output schema and vocabulary validation."
    },
    {
      "metric_id": "core_exact_match_rate",
      "level": "all_samples",
      "definition": "Fraction of samples where task_type, primary_issue, severity, evidence_codes set, and recommended_action_codes set all exactly match Ground Truth. Explanation is excluded."
    },
    {
      "metric_id": "primary_issue_accuracy",
      "level": "experiment_diagnosis_only",
      "definition": "Fraction of experiment_diagnosis samples whose primary_issue exactly matches Ground Truth."
    },
    {
      "metric_id": "primary_issue_macro_f1",
      "level": "experiment_diagnosis_only",
      "definition": "Unweighted mean of per-class F1 over diagnosis primary_issue classes."
    },
    {
      "metric_id": "severity_accuracy",
      "level": "experiment_diagnosis_only",
      "definition": "Fraction of experiment_diagnosis samples whose severity exactly matches Ground Truth."
    },
    {
      "metric_id": "evidence_exact_set_accuracy",
      "level": "all_samples",
      "definition": "Fraction of samples whose unordered evidence_codes set exactly matches Ground Truth."
    },
    {
      "metric_id": "evidence_micro_f1",
      "level": "all_samples",
      "definition": "Micro-averaged multilabel F1 over canonical evidence codes."
    },
    {
      "metric_id": "recommendation_exact_set_accuracy",
      "level": "all_samples",
      "definition": "Fraction of samples whose unordered recommended_action_codes set exactly matches Ground Truth."
    },
    {
      "metric_id": "recommendation_micro_f1",
      "level": "all_samples",
      "definition": "Micro-averaged multilabel F1 over canonical recommended action codes."
    },
    {
      "metric_id": "structural_hallucination_rate",
      "level": "all_samples",
      "definition": "Fraction of responses containing out-of-vocabulary categorical values, unknown evidence/action codes, or prohibited extra output fields."
    }
  ],
  "task_slice_metrics": {
    "experiment_diagnosis": [
      "core_exact_match_rate",
      "primary_issue_accuracy",
      "primary_issue_macro_f1",
      "severity_accuracy",
      "evidence_exact_set_accuracy",
      "evidence_micro_f1",
      "recommendation_exact_set_accuracy",
      "recommendation_micro_f1"
    ],
    "metric_interpretation": [
      "core_exact_match_rate",
      "evidence_exact_set_accuracy",
      "evidence_micro_f1",
      "recommendation_exact_set_accuracy",
      "recommendation_micro_f1"
    ],
    "model_comparison": [
      "core_exact_match_rate",
      "evidence_exact_set_accuracy",
      "evidence_micro_f1",
      "recommendation_exact_set_accuracy",
      "recommendation_micro_f1"
    ]
  },
  "hard_test_metrics": [
    {
      "metric_id": "hard_core_exact_match_rate",
      "definition": "Core Exact Match Rate across all Hard Test samples."
    },
    {
      "metric_id": "hard_group_core_exact_match_rate",
      "definition": "Core Exact Match Rate reported separately for each Hard Test Group."
    },
    {
      "metric_id": "hard_family_core_exact_match_rate",
      "definition": "Core Exact Match Rate reported separately for each Hard Test Family."
    },
    {
      "metric_id": "directional_pair_success_rate",
      "property_type": "directional_boundary",
      "unit": "pair",
      "definition": "Fraction of directional pairs for which both pair members have the correct primary_issue, therefore satisfying the expected threshold-crossing relation."
    },
    {
      "metric_id": "invariance_consistency_rate",
      "property_type": "invariance_distractor",
      "unit": "pair",
      "definition": "Fraction of invariance pairs whose predicted decision signatures are identical between base and perturbed inputs, regardless of correctness."
    },
    {
      "metric_id": "invariance_correct_pair_rate",
      "property_type": "invariance_distractor",
      "unit": "pair",
      "definition": "Fraction of invariance pairs where both predictions match the Ground Truth decision signature and remain identical across the perturbation."
    },
    {
      "metric_id": "invariance_violation_rate",
      "property_type": "invariance_distractor",
      "unit": "pair",
      "definition": "Fraction of invariance pairs whose predicted decision signatures differ between base and perturbed inputs."
    },
    {
      "metric_id": "priority_composition_accuracy",
      "property_type": "priority_composition",
      "unit": "sample",
      "definition": "Primary Issue Accuracy on Priority Composition samples."
    },
    {
      "metric_id": "hard_json_schema_valid_rate",
      "definition": "JSON Schema Valid Rate across all Hard Test samples."
    }
  ],
  "pair_metric_policy": {
    "directional_pair_requires_both_members": true,
    "invariance_pair_requires_both_members": true,
    "pair_with_parse_failure_counts_as_pair_failure_for_correct_pair_metrics": true,
    "pair_with_one_parse_failure_cannot_count_as_consistent": true,
    "invariance_violation_rate_equals_one_minus_consistency_rate": true
  },
  "failure_policy": {
    "parse_failure": {
      "parse_success": 0,
      "schema_valid": 0,
      "core_exact_match": 0,
      "field_level_correctness": 0
    },
    "schema_invalid_but_parseable": {
      "parse_success": 1,
      "schema_valid": 0,
      "core_exact_match": 0
    },
    "missing_required_field": {
      "schema_valid": 0,
      "core_exact_match": 0
    }
  },
  "deferred_metrics": [
    {
      "metric_id": "free_text_explanation_quality",
      "reason": "Requires a separate human rubric or validated judge; exact string matching is inappropriate."
    },
    {
      "metric_id": "numeric_faithfulness",
      "reason": "Reserved for the later Tool Calling Agent evaluation where explicit tool-returned numeric values are available."
    }
  ]
}
