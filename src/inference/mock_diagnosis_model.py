"""Mock diagnosis model used only by the Phase 1 Agent skeleton."""

from typing import Any


class MockDiagnosisModel:
    """Return a deterministic diagnosis without loading model weights."""

    def diagnose(
        self,
        structured_context: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(structured_context, dict):
            raise TypeError("structured_context must be a dictionary")

        return {
            "task_type": "experiment_diagnosis",
            "primary_issue": "overfitting",
            "severity": "medium",
            "evidence_codes": ["mock_evidence"],
            "recommended_action_codes": ["mock_action"],
            "explanation": "mock diagnosis",
        }
