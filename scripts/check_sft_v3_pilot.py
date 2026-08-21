import json
from collections import Counter
from pathlib import Path


PATH = Path(
    "data/generated/sft_v3_pilot.jsonl"
)


EXPECTED = {

    "priority_composition":2000,

    "directional_boundary":1200,

    "output_completion":800,

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


    counter=Counter(
        x["category"]
        for x in records
    )


    errors=[]


    if len(records)!=4000:

        errors.append(
            "sample count error"
        )


    if dict(counter)!=EXPECTED:

        errors.append(
            str(counter)
        )


    for item in records:

        if len(
            item["messages"]
        ) != 3:

            errors.append(
                "message format error"
            )

            break


        json.loads(
            item["messages"][2]["content"]
        )


    if errors:

        print(
            "SFT V3 PILOT VALIDATION FAILED"
        )

        for e in errors:

            print(
                "-",
                e
            )

        raise SystemExit(1)



    print(
        "SFT V3 PILOT VALIDATION PASSED"
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
