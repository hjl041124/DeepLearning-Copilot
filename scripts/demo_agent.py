"""Command-line demo for the DeepLearning-Copilot Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.service import run_diagnosis  # noqa: E402


REQUIRED_INPUT_SECTIONS = (
    "experiment_context",
    "metrics",
    "training_log",
    "dataset_info",
)


def load_demo_input(input_path: str | Path) -> dict[str, Any]:
    """Read and minimally validate the demo JSON input."""

    path = Path(input_path)
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("demo input must be a JSON object")

    missing = [
        section
        for section in REQUIRED_INPUT_SECTIONS
        if section not in payload
    ]
    if missing:
        raise ValueError(
            "demo input is missing required sections: "
            + ", ".join(missing)
        )

    return payload


def format_diagnosis_output(agent_result: dict[str, Any]) -> str:
    """Format diagnosis fields for display without changing them."""

    diagnosis = agent_result.get("diagnosis")
    if not isinstance(diagnosis, dict):
        error = agent_result.get("error") or "diagnosis is unavailable"
        raise ValueError(f"Agent diagnosis failed: {error}")

    return "\n".join(
        [
            f"Experiment ID: {agent_result.get('experiment_id', '')}",
            f"Task Type: {diagnosis.get('task_type', '')}",
            f"Primary Issue: {diagnosis.get('primary_issue', '')}",
            f"Severity: {diagnosis.get('severity', '')}",
            "Evidence Codes: "
            + ", ".join(diagnosis.get("evidence_codes", [])),
            "Recommended Action Codes: "
            + ", ".join(
                diagnosis.get("recommended_action_codes", [])
            ),
            f"Explanation: {diagnosis.get('explanation', '')}",
        ]
    )


def run_demo(
    input_path: str | Path,
    diagnosis_runner: Callable[
        [str, dict[str, Any]],
        dict[str, Any],
    ]
    | None = None,
) -> str:
    """Read one demo experiment, run the Agent, and format the result."""

    payload = load_demo_input(input_path)
    experiment_id = payload.get("experiment_id") or Path(input_path).stem
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")

    user_input = {
        key: value
        for key, value in payload.items()
        if key != "experiment_id"
    }
    runner = diagnosis_runner or run_diagnosis
    result = runner(experiment_id, user_input)
    return format_diagnosis_output(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the DeepLearning-Copilot Agent demo."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a structured experiment JSON file.",
    )
    args = parser.parse_args(argv)

    try:
        output = run_demo(args.input)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
