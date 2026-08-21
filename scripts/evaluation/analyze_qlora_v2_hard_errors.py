import json
from pathlib import Path
from collections import defaultdict, Counter


INPUT = Path(
    "reports/qlora_v2/hard_test_predictions.jsonl"
)


OUTPUT_JSON = Path(
    "reports/qlora_v2/hard_error_analysis.json"
)


OUTPUT_MD = Path(
    "reports/qlora_v2/hard_error_analysis.md"
)


def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                data.append(
                    json.loads(line)
                )

    return data



def parse_prediction(text):

    try:

        start = text.find("{")
        end = text.rfind("}")

        if start == -1:
            return None

        return json.loads(
            text[start:end+1]
        )

    except:

        return None



def main():

    records = load_jsonl(
        INPUT
    )


    family_stats = defaultdict(
        lambda:{
            "total":0,
            "core_correct":0,
            "issue_correct":0,
        }
    )


    issue_stats = Counter()


    property_stats = defaultdict(
        lambda:{
            "total":0,
            "core_correct":0,
        }
    )


    for item in records:

        pred = parse_prediction(
            item["prediction"]
        )

        gt = item["ground_truth"]


        family = item.get(
            "hard_test_family_id",
            "unknown"
        )


        prop = item.get(
            "test_property_type",
            "unknown"
        )


        family_stats[family]["total"] += 1

        property_stats[prop]["total"] += 1


        if pred is None:
            continue


        issue_ok = (
            pred.get(
                "primary_issue"
            )
            ==
            gt.get(
                "primary_issue"
            )
        )


        if issue_ok:

            family_stats[family][
                "issue_correct"
            ] += 1

            issue_stats[
                gt.get("primary_issue")
            ] += 1



        core_ok = (

            pred.get("task_type")
            ==
            gt.get("task_type")

            and

            pred.get("primary_issue")
            ==
            gt.get("primary_issue")

            and

            pred.get("severity")
            ==
            gt.get("severity")

            and

            set(
                pred.get(
                    "evidence_codes",
                    []
                )
            )
            ==
            set(
                gt.get(
                    "evidence_codes",
                    []
                )
            )

            and

            set(
                pred.get(
                    "recommended_action_codes",
                    []
                )
            )
            ==
            set(
                gt.get(
                    "recommended_action_codes",
                    []
                )
            )
        )


        if core_ok:

            family_stats[family][
                "core_correct"
            ] += 1

            property_stats[prop][
                "core_correct"
            ] += 1



    result = {

        "family_stats":
            dict(family_stats),

        "property_stats":
            dict(property_stats),

        "issue_correct_distribution":
            dict(issue_stats),

    }


    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    md=[]

    md.append(
        "# QLoRA v2 Hard Test Error Analysis\n\n"
    )


    md.append(
        "## Family Statistics\n\n"
    )

    for k,v in family_stats.items():

        md.append(
            f"- {k}: {v}\n"
        )


    md.append(
        "\n## Property Statistics\n\n"
    )


    for k,v in property_stats.items():

        md.append(
            f"- {k}: {v}\n"
        )


    OUTPUT_MD.write_text(
        "".join(md),
        encoding="utf-8",
    )


    print(
        "QLORA V2 HARD ERROR ANALYSIS PASSED"
    )


    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
