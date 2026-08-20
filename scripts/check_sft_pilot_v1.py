import json
from collections import Counter
from pathlib import Path


PATH = Path(
    "data/generated/sft_pilot_v1.jsonl"
)


EXPECTED = {
    "schema_alignment":800,
    "evidence_grounding":600,
    "recommendation_mapping":400,
    "hard_reasoning":200,
}


def main():

    records=[]

    with open(
        PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            records.append(
                json.loads(line)
            )


    errors=[]


    if len(records)!=2000:

        errors.append(
            "sample count error"
        )


    counter=Counter(
        x["category"]
        for x in records
    )


    if dict(counter)!=EXPECTED:

        errors.append(
            f"category error {counter}"
        )


    for item in records:

        if len(
            item["messages"]
        )!=3:

            errors.append(
                "message format error"
            )
            break


        if (
            item["messages"][2]["role"]
            !=
            "assistant"
        ):

            errors.append(
                "assistant role error"
            )
            break


        try:

            json.loads(
                item["messages"][2]["content"]
            )

        except:

            errors.append(
                "assistant json error"
            )
            break


    if errors:

        print(
            "SFT PILOT VALIDATION FAILED"
        )

        for e in errors:

            print(
                "-",
                e
            )

        raise SystemExit(1)


    print(
        "SFT PILOT VALIDATION PASSED"
    )

    print(
        "Samples:",
        len(records)
    )

    print(
        "Distribution:",
        dict(counter)
    )


if __name__=="__main__":
    main()
