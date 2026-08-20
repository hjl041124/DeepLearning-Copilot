import json
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.diagnosis_pipeline import diagnose_record
from src.evaluation.output_validator import validate_output


ROOT = Path(__file__).resolve().parents[2]

MAPPING_PATH = (
    ROOT / "configs" / "recommendation_mapping_v1.json"
)

with MAPPING_PATH.open(encoding="utf-8") as f:
    MAPPING = json.load(f)


def _unique(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def build_recommendations(
    primary_issue: str,
    evidence_codes: List[str],
) -> List[str]:

    actions = list(
        MAPPING["default_actions"].get(
            primary_issue,
            [],
        )
    )

    evidence_mapping = MAPPING[
        "evidence_specific_actions"
    ]

    for code in evidence_codes:
        actions.extend(
            evidence_mapping.get(code, [])
        )

    return _unique(actions)


def build_explanation(
    primary_issue: str,
    evidence_codes: List[str],
) -> str:

    if primary_issue == "no_clear_issue":
        return (
            "No strong diagnostic signal was detected "
            "from the provided experiment evidence."
        )

    evidence_text = ", ".join(evidence_codes)

    return (
        f"The primary diagnosis is {primary_issue}. "
        f"The decision is supported by: {evidence_text}."
    )


def build_ground_truth(
    record: Dict[str, Any],
) -> Dict[str, Any]:

    pipeline_result = diagnose_record(record)

    diagnosis = pipeline_result["diagnosis"]

    primary_issue = diagnosis["primary_issue"]
    severity = diagnosis["severity"]
    evidence_codes = diagnosis["evidence_codes"]

    output = {
        "task_type": "experiment_diagnosis",
        "primary_issue": primary_issue,
        "severity": severity,
        "evidence_codes": evidence_codes,
        "recommended_action_codes": build_recommendations(
            primary_issue,
            evidence_codes,
        ),
        "explanation": build_explanation(
            primary_issue,
            evidence_codes,
        ),
    }

    errors = validate_output(output)

    if errors:
        raise ValueError(
            "Generated ground truth failed validation: "
            + "; ".join(errors)
        )

    return {
        "features": pipeline_result["features"],
        "ground_truth": output,
    }
