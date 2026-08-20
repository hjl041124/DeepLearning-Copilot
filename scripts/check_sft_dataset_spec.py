import json
from pathlib import Path


path = Path(
    "configs/sft_dataset_spec_v1.json"
)


data = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)


errors = []


if data["version"] != "1.0":
    errors.append(
        "version error"
    )


composition = data["composition"]


if abs(
    sum(composition.values())
    - 1.0
) > 1e-6:

    errors.append(
        "composition ratio error"
    )


required = {
    "schema_alignment",
    "evidence_grounding",
    "recommendation_mapping",
    "hard_reasoning"
}


if set(composition) != required:

    errors.append(
        "composition keys error"
    )


if (
    data["split_policy"]["random_sample_split"]
    is True
):

    errors.append(
        "random split forbidden"
    )


if (
    data["split_policy"]["split_unit"]
    != "template_family_id"
):

    errors.append(
        "split unit must be template_family_id"
    )


if errors:

    print(
        "SFT DATASET SPEC VALIDATION FAILED"
    )

    for e in errors:
        print("-", e)

    raise SystemExit(1)


print(
    "SFT DATASET SPEC VALIDATION PASSED"
)

print(
    "Target samples:",
    data["target_samples"]
)

print(
    "Composition:",
    composition
)
