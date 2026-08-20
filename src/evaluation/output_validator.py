import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]

SCHEMA_PATH = ROOT / "configs" / "output_schema_v1.json"
VOCAB_PATH = ROOT / "configs" / "output_vocabulary_v1.json"


with SCHEMA_PATH.open(encoding="utf-8") as f:
    SCHEMA = json.load(f)

with VOCAB_PATH.open(encoding="utf-8") as f:
    VOCAB = json.load(f)


VALID_EVIDENCE = set(VOCAB["evidence_codes"])
VALID_ACTIONS = set(VOCAB["recommended_action_codes"])

SCHEMA_VALIDATOR = Draft202012Validator(SCHEMA)


def validate_output(output: dict) -> list[str]:
    errors = []

    for error in SCHEMA_VALIDATOR.iter_errors(output):
        errors.append(f"schema: {error.message}")

    for code in output.get("evidence_codes", []):
        if code not in VALID_EVIDENCE:
            errors.append(f"unknown evidence code: {code}")

    for code in output.get("recommended_action_codes", []):
        if code not in VALID_ACTIONS:
            errors.append(f"unknown action code: {code}")

    return errors
