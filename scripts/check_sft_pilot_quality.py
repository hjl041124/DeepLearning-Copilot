import json
from collections import Counter
from pathlib import Path


PATH = Path(
    "data/generated/sft_pilot_v1.jsonl"
)


ALLOWED_ISSUES = {
    "overfitting",
    "underfitting",
    "optimization_problem",
    "class_imbalance",
    "data_quality_issue",
    "no_clear_issue",
    "not_applicable",
}


REQUIRED_FIELDS = {
    "task_type",
    "primary_issue",
    "severity",
    "evidence_codes",
    "recommended_action_codes",
    "explanation",
}


def load_data():

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

    return records



def main():

    records = load_data()


    errors=[]


    issue_counter = Counter()

    evidence_counter = Counter()

    action_counter = Counter()


    natural_language_issue = 0


    for item in records:


        try:

            output = json.loads(
                item[
                    "messages"
                ][2][
                    "content"
                ]
            )

        except:

            errors.append(
                "assistant json parse error"
            )

            continue



        if set(output.keys()) != REQUIRED_FIELDS:

            errors.append(
                "output field mismatch"
            )


        issue = output.get(
            "primary_issue"
        )


        issue_counter[
            str(issue)
        ] += 1



        if (
            issue
            not in
            ALLOWED_ISSUES
        ):

            natural_language_issue += 1



        for code in output.get(
            "evidence_codes",
            []
        ):

            evidence_counter[
                code
            ] += 1



        for code in output.get(
            "recommended_action_codes",
            []
        ):

            action_counter[
                code
            ] += 1



    if natural_language_issue:

        errors.append(
            f"found {natural_language_issue} invalid primary_issue labels"
        )



    if errors:

        print(
            "SFT PILOT QUALITY CHECK FAILED"
        )

        for e in errors:

            print(
                "-",
                e
            )

        raise SystemExit(1)



    print(
        "SFT PILOT QUALITY CHECK PASSED"
    )


    print()

    print(
        "Primary Issue Distribution:"
    )

    print(
        dict(issue_counter)
    )


    print()

    print(
        "Evidence Code Count:"
    )

    print(
        len(evidence_counter)
    )


    print()

    print(
        "Recommendation Code Count:"
    )

    print(
        len(action_counter)
    )


if __name__ == "__main__":
    main()
