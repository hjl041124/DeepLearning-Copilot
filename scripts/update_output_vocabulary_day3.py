import json
from pathlib import Path


ROOT = Path.cwd()
PATH = ROOT / "configs" / "output_vocabulary_v1.json"


NEW_EVIDENCE_CODES = [
    "precision_dominates_recall",
    "recall_dominates_precision",
    "model_a_higher_primary_metric",
    "model_b_higher_primary_metric",
    "model_a_lower_latency",
    "model_b_lower_latency",
    "quality_efficiency_tradeoff",
    "model_a_higher_macro_f1",
    "model_b_higher_macro_f1",
    "no_material_model_difference"
]


NEW_ACTION_CODES = [
    "inspect_false_positives",
    "inspect_false_negatives",
    "inspect_worst_class",
    "inspect_generalization_gap",
    "prefer_model_a",
    "prefer_model_b",
    "choose_by_deployment_constraint",
    "prioritize_macro_f1",
    "collect_more_evaluation_evidence"
]


def append_unique(target, values):
    for value in values:
        if value not in target:
            target.append(value)


def main():
    if not PATH.exists():
        raise SystemExit(
            f"UPDATE FAILED: missing file: {PATH}"
        )

    data = json.loads(PATH.read_text(encoding="utf-8"))

    evidence = data.get("evidence_codes")
    actions = data.get("recommended_action_codes")

    if not isinstance(evidence, list):
        raise SystemExit(
            "UPDATE FAILED: evidence_codes must be a list"
        )

    if not isinstance(actions, list):
        raise SystemExit(
            "UPDATE FAILED: recommended_action_codes must be a list"
        )

    append_unique(evidence, NEW_EVIDENCE_CODES)
    append_unique(actions, NEW_ACTION_CODES)

    PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    print("OUTPUT VOCABULARY UPDATE PASSED")
    print("evidence_codes:", len(evidence))
    print("recommended_action_codes:", len(actions))


if __name__ == "__main__":
    main()
