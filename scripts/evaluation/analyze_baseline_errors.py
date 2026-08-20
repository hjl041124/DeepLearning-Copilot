import json
from collections import Counter, defaultdict
from pathlib import Path


BASELINE_DIR = Path(
    "reports/baseline"
)


OUTPUT_JSON = (
    BASELINE_DIR
    /
    "baseline_error_analysis.json"
)

OUTPUT_MD = (
    BASELINE_DIR
    /
    "baseline_error_analysis.md"
)


ALLOWED_TASK_TYPES = {
    "experiment_diagnosis",
    "metric_interpretation",
    "model_comparison",
}


def load_jsonl(path):

    records = []

    with open(
        path,
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



def parse_prediction(text):

    try:

        start = text.find("{")
        end = text.rfind("}")

        if start == -1:
            return None

        return json.loads(
            text[start:end+1]
        )

    except Exception:
        return None



def analyze_file(path):

    records = load_jsonl(
        path
    )

    stats = {

        "total_samples":
            len(records),

        "parse_fail":
            0,

        "task_type_error":
            0,

        "primary_issue_error":
            0,

        "evidence_error":
            0,

        "recommendation_error":
            0,

        "complete_match_error":
            0,

        "prediction_primary_issue_distribution":
            Counter(),

        "ground_truth_primary_issue_distribution":
            Counter(),

        "hard_property_distribution":
            Counter(),

        "hard_property_accuracy":
            defaultdict(
                lambda: {
                    "total": 0,
                    "correct": 0
                }
            ),
    }


    for item in records:

        pred = parse_prediction(
            item["prediction"]
        )

        gt = item[
            "ground_truth"
        ]


        if pred is None:

            stats[
                "parse_fail"
            ] += 1

            continue


        pred_issue = pred.get(
            "primary_issue"
        )

        gt_issue = gt.get(
            "primary_issue"
        )


        stats[
            "prediction_primary_issue_distribution"
        ][
            str(pred_issue)
        ] += 1


        stats[
            "ground_truth_primary_issue_distribution"
        ][
            str(gt_issue)
        ] += 1



        if (
            pred.get(
                "task_type"
            )
            !=
            gt.get(
                "task_type"
            )
        ):

            stats[
                "task_type_error"
            ] += 1



        if (
            pred_issue
            !=
            gt_issue
        ):

            stats[
                "primary_issue_error"
            ] += 1



        if (
            set(
                pred.get(
                    "evidence_codes",
                    []
                )
            )
            !=
            set(
                gt.get(
                    "evidence_codes",
                    [])
            )
        ):

            stats[
                "evidence_error"
            ] += 1



        if (
            set(
                pred.get(
                    "recommended_action_codes",
                    []
                )
            )
            !=
            set(
                gt.get(
                    "recommended_action_codes",
                    [])
            )
        ):

            stats[
                "recommendation_error"
            ] += 1



        complete = (
            pred.get(
                "task_type"
            )
            ==
            gt.get(
                "task_type"
            )
            and
            pred.get(
                "primary_issue"
            )
            ==
            gt.get(
                "primary_issue"
            )
            and
            pred.get(
                "severity"
            )
            ==
            gt.get(
                "severity"
            )
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
                    [])
            )
            and
            set(
                pred.get(
                    "recommended_action_codes",
                    [])
            )
            ==
            set(
                gt.get(
                    "recommended_action_codes",
                    [])
            )
        )


        if not complete:

            stats[
                "complete_match_error"
            ] += 1



        if (
            "hard_test_group_id"
            in item
        ):

            prop = item[
                "test_property_type"
            ]

            stats[
                "hard_property_distribution"
            ][
                prop
            ] += 1


    return stats



def convert_counter(obj):

    if isinstance(
        obj,
        Counter
    ):
        return dict(obj)

    if isinstance(
        obj,
        defaultdict
    ):
        return dict(obj)

    return obj



def clean(obj):

    if isinstance(
        obj,
        dict
    ):

        return {
            k: clean(v)
            for k, v in obj.items()
        }


    if isinstance(
        obj,
        Counter
    ):

        return dict(obj)


    if isinstance(
        obj,
        defaultdict
    ):

        return dict(obj)


    return obj



def main():

    results = {}


    for name in [
        "standard_test",
        "hard_test",
    ]:

        results[name] = clean(
            analyze_file(
                BASELINE_DIR
                /
                f"{name}_predictions.jsonl"
            )
        )


    OUTPUT_JSON.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    md = []

    md.append(
        "# Baseline Error Analysis\n\n"
    )

    md.append(
        "Model: Qwen3-4B-Instruct-2507 Base Model\n\n"
    )


    for name, data in results.items():

        md.append(
            f"## {name}\n\n"
        )


        for key, value in data.items():

            md.append(
                f"### {key}\n\n"
            )

            md.append(
                f"{value}\n\n"
            )


    OUTPUT_MD.write_text(
        "".join(md),
        encoding="utf-8",
    )


    print(
        "BASELINE ERROR ANALYSIS PASSED"
    )


    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
