import json
from pathlib import Path


ROOT = Path.cwd()

DATA_PATH = (
    ROOT
    / "data"
    / "generated"
    / "pilot_dataset_v1.jsonl"
)


def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def main():
    records = load_jsonl(
        DATA_PATH
    )

    selected = {}

    for record in records:
        presentation = record[
            "presentation_family_id"
        ]

        if presentation not in selected:
            selected[
                presentation
            ] = record

    for index, presentation in enumerate(
        sorted(selected),
        start=1,
    ):
        record = selected[
            presentation
        ]

        print()
        print(
            "=" * 80
        )

        print(
            f"PREVIEW {index}"
        )

        print(
            "Presentation Family（呈现结构族）：",
            presentation,
        )

        print(
            "Split（数据划分）：",
            record["split"],
        )

        print(
            "Task Type（任务类型）：",
            record["task_type"],
        )

        print(
            "Scenario Family（场景族）：",
            record[
                "scenario_family_id"
            ],
        )

        print()
        print(
            "--- PROMPT（提示词） ---"
        )
        print(
            record[
                "prompt"
            ]
        )

        print()
        print(
            "--- GROUND TRUTH（标准答案） ---"
        )

        print(
            json.dumps(
                record[
                    "ground_truth"
                ],
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
