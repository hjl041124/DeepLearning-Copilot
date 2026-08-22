"""Parse and validate structured diagnosis model output."""

import json
from dataclasses import dataclass
from typing import Any

from src.evaluation.output_validator import validate_output
from src.inference.semantic_alignment import align_model_output


@dataclass
class ParsedModelOutput:
    """Model output together with its unmodified raw representation."""

    raw_model_output: str
    diagnosis: dict[str, Any] | None
    validation_errors: list[str]

    @property
    def is_valid(self) -> bool:
        return self.diagnosis is not None and not self.validation_errors


def parse_model_output(raw_model_output: str) -> ParsedModelOutput:
    """Extract, deterministically align, and validate one JSON object."""

    if not isinstance(raw_model_output, str):
        return ParsedModelOutput(
            raw_model_output=str(raw_model_output),
            diagnosis=None,
            validation_errors=["model output must be a string"],
        )

    start = raw_model_output.find("{")
    end = raw_model_output.rfind("}")

    if start == -1 or end == -1 or end < start:
        return ParsedModelOutput(
            raw_model_output=raw_model_output,
            diagnosis=None,
            validation_errors=["no JSON object found in model output"],
        )

    try:
        diagnosis = json.loads(raw_model_output[start : end + 1])
    except json.JSONDecodeError as exc:
        return ParsedModelOutput(
            raw_model_output=raw_model_output,
            diagnosis=None,
            validation_errors=[f"invalid JSON: {exc.msg}"],
        )

    if not isinstance(diagnosis, dict):
        return ParsedModelOutput(
            raw_model_output=raw_model_output,
            diagnosis=None,
            validation_errors=["diagnosis output must be a JSON object"],
        )

    diagnosis = align_model_output(diagnosis)

    return ParsedModelOutput(
        raw_model_output=raw_model_output,
        diagnosis=diagnosis,
        validation_errors=validate_output(diagnosis),
    )
