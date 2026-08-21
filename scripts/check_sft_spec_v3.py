import json
from pathlib import Path


path = Path(
    "configs/sft_dataset_spec_v3.json"
)


data = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)


errors=[]


composition = data["composition"]


if abs(
    sum(composition.values()) - 1.0
) > 1e-6:

    errors.append(
        "composition sum error"
    )


if (
    composition["priority_composition"]
    != 0.5
):

    errors.append(
        "priority ratio error"
    )


if (
    composition["directional_boundary"]
    != 0.3
):

    errors.append(
        "boundary ratio error"
    )


if (
    data["split_policy"]["random_split"]
):

    errors.append(
        "random split forbidden"
    )


if errors:

    print(
        "SFT V3 SPEC VALIDATION FAILED"
    )

    for e in errors:
        print("-", e)

    raise SystemExit(1)


print(
    "SFT V3 SPEC VALIDATION PASSED"
)

print(
    "Target samples:",
    data["target_samples"]
)

print(
    "Composition:",
    composition
)
