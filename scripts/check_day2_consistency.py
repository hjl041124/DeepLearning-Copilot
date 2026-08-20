import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"


def load(name):
    with (CONFIG / name).open("r", encoding="utf-8") as f:
        return json.load(f)


taxonomy = load("taxonomy_v1.json")
rules = load("diagnosis_rules_v1.json")
features = load("feature_definitions_v1.json")
thresholds = load("threshold_bands_v1.json")
schema = load("output_schema_v1.json")
vocab = load("output_vocabulary_v1.json")
recommendations = load("recommendation_mapping_v1.json")


errors = []

# --------------------------------------------------
# Taxonomy（分類體系）
# --------------------------------------------------

task_types = {
    x["id"]
    for x in taxonomy["task_types"]
}

primary_issues = {
    x["id"]
    for x in taxonomy["primary_issues"]
}

schema_task_types = set(
    schema["properties"]["task_type"]["enum"]
)

schema_primary_issues = set(
    schema["properties"]["primary_issue"]["enum"]
)

if task_types != schema_task_types:
    errors.append(
        f"Task type mismatch: "
        f"taxonomy={task_types}, schema={schema_task_types}"
    )

if primary_issues != schema_primary_issues:
    errors.append(
        f"Primary issue mismatch: "
        f"taxonomy={primary_issues}, schema={schema_primary_issues}"
    )


# --------------------------------------------------
# Diagnosis rules（診斷規則）
# --------------------------------------------------

rule_issues = set(rules["rules"].keys())

expected_rule_issues = {
    x
    for x in primary_issues
    if x != "not_applicable"
}

if rule_issues != expected_rule_issues:
    errors.append(
        f"Rule issue mismatch: "
        f"rules={rule_issues}, expected={expected_rule_issues}"
    )


# --------------------------------------------------
# Recommendation mapping（建議映射）
# --------------------------------------------------

mapped_issues = set(
    recommendations["default_actions"].keys()
)

if mapped_issues != expected_rule_issues:
    errors.append(
        f"Recommendation issue mismatch: "
        f"mapping={mapped_issues}, expected={expected_rule_issues}"
    )


# --------------------------------------------------
# Evidence vocabulary（證據詞彙）
# --------------------------------------------------

valid_evidence = set(vocab["evidence_codes"])
valid_actions = set(vocab["recommended_action_codes"])

for evidence_code, actions in (
    recommendations["evidence_specific_actions"].items()
):
    if evidence_code not in valid_evidence:
        errors.append(
            f"Unknown evidence in recommendation mapping: "
            f"{evidence_code}"
        )

    for action in actions:
        if action not in valid_actions:
            errors.append(
                f"Unknown action in recommendation mapping: "
                f"{action}"
            )


for issue, actions in recommendations["default_actions"].items():
    for action in actions:
        if action not in valid_actions:
            errors.append(
                f"Unknown default action for {issue}: {action}"
            )


# --------------------------------------------------
# Feature configuration（特徵配置）
# --------------------------------------------------

feature_names = set(features["features"].keys())

required_features = {
    "relative_generalization_gap",
    "late_degradation",
    "relative_improvement",
    "plateau_streak",
    "oscillation_score",
    "relative_amplitude",
    "class_imbalance_ratio",
    "class_performance_gap",
    "class_performance_ratio",
    "accuracy_macro_f1_gap",
    "reference_performance_gap",
    "label_noise_rate",
    "duplicate_rate",
    "missing_value_rate",
    "corrupted_sample_rate",
    "split_overlap_rate",
}

missing_features = required_features - feature_names

if missing_features:
    errors.append(
        f"Missing feature definitions: {missing_features}"
    )


# --------------------------------------------------
# Threshold configuration（閾值配置）
# --------------------------------------------------

required_threshold_sections = {
    "generalization",
    "underfitting",
    "class_imbalance",
    "optimization",
    "data_quality",
}

missing_threshold_sections = (
    required_threshold_sections
    - set(thresholds.keys())
)

if missing_threshold_sections:
    errors.append(
        f"Missing threshold sections: "
        f"{missing_threshold_sections}"
    )


# --------------------------------------------------
# Result（結果）
# --------------------------------------------------

print("=== Day 2 Consistency Check ===")
print("Task types:", len(task_types))
print("Primary issues:", len(primary_issues))
print("Feature definitions:", len(feature_names))
print("Evidence codes:", len(valid_evidence))
print("Action codes:", len(valid_actions))

if errors:
    print("\nCONSISTENCY CHECK FAILED")

    for error in errors:
        print("-", error)

    raise SystemExit(1)

print("\nDAY2 CONFIG CONSISTENCY CHECK PASSED")
