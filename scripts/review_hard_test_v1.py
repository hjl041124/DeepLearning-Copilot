import json
from pathlib import Path


ROOT = Path.cwd()

DATA_PATH = (
    ROOT
    / "data"
    / "generated"
    / "hard_test_v1.jsonl"
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
        family_id = record[
            "hard_test_family_id"
        ]

        if (
            family_id
            not in selected
        ):
            selected[
                family_id
            ] = record

    for index, family_id in enumerate(
        sorted(
            selected
        ),
        start=1,
    ):
        record = selected[
            family_id
        ]

        print()
        print(
            "=" * 90
        )

        print(
            f"HARD TEST FAMILY {index}"
        )

        print(
            "Hard Test Family（困难测试族）：",
            family_id,
        )

        print(
            "Test Property（测试属性）：",
            record[
                "test_property_type"
            ],
        )

        print(
            "Pair ID（样本对 ID）：",
            record[
                "pair_id"
            ],
        )

        print(
            "Pair Member（样本对成员）：",
            record[
                "pair_member"
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

        print()
        print(
            "--- PROPERTY METADATA（测试属性元数据） ---"
        )

        print(
            json.dumps(
                record[
                    "property_metadata"
                ],
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
