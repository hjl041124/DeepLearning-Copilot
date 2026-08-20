import json
from pathlib import Path


path = Path(
    "configs/sft_dataset_spec_v2.json"
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


hard = data[
    "hard_reasoning_breakdown"
]


if abs(
    sum(hard.values()) - 0.3
) > 1e-6:

    errors.append(
        "hard reasoning ratio error"
    )


if (
    data["split_policy"]["random_split"]
):

    errors.append(
        "random split forbidden"
    )


if errors:

    print(
        "SFT V2 SPEC VALIDATION FAILED"
    )

    for e in errors:
        print("-", e)

    raise SystemExit(1)


print(
    "SFT V2 SPEC VALIDATION PASSED"
)

print(
    "Target samples:",
    data["target_samples"]
)

print(
    "Composition:",
    composition
)

print(
    "Hard reasoning:",
    hard
)
