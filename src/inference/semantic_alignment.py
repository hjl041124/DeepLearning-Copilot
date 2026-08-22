"""Deterministic alignment for confirmed model-output aliases."""

from copy import deepcopy
import re
from typing import Any


PRIMARY_ISSUE_ALIASES = {
    "class_imbalance_issue": "class_imbalance",
}

EVIDENCE_CODE_ALIASES = {
    "small_majority_class_f1": "large_class_performance_gap",
}


def _deduplicate(values: Any) -> Any:
    """Remove duplicate list items while preserving their original order."""

    if not isinstance(values, list):
        return values

    deduplicated = []
    for value in values:
        if value not in deduplicated:
            deduplicated.append(value)
    return deduplicated


def _replace_exact_token(text: Any, alias: str, canonical: str) -> Any:
    """Replace a standalone alias in explanation text."""

    if not isinstance(text, str):
        return text

    pattern = rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])"
    return re.sub(pattern, canonical, text)


def align_model_output(output: dict[str, Any]) -> dict[str, Any]:
    """Apply only approved aliases and deterministic array cleanup.

    Unknown values are deliberately preserved so that the existing output
    validator can reject them.
    """

    aligned = deepcopy(output)
    primary_issue = aligned.get("primary_issue")

    if primary_issue in PRIMARY_ISSUE_ALIASES:
        canonical = PRIMARY_ISSUE_ALIASES[primary_issue]
        aligned["primary_issue"] = canonical
        aligned["explanation"] = _replace_exact_token(
            aligned.get("explanation"),
            primary_issue,
            canonical,
        )

    evidence_codes = aligned.get("evidence_codes")
    if isinstance(evidence_codes, list):
        mapped_evidence_codes = []
        for code in evidence_codes:
            canonical = EVIDENCE_CODE_ALIASES.get(code, code)
            mapped_evidence_codes.append(canonical)

            if canonical != code:
                aligned["explanation"] = _replace_exact_token(
                    aligned.get("explanation"),
                    code,
                    canonical,
                )

        aligned["evidence_codes"] = mapped_evidence_codes

    for field in ("evidence_codes", "recommended_action_codes"):
        if field in aligned:
            aligned[field] = _deduplicate(aligned[field])

    return aligned
