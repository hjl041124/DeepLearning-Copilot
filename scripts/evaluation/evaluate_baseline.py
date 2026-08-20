import json
from pathlib import Path
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


BASELINE_DIR = Path(
    "reports/baseline"
)


OUTPUT_DIR = Path(
    "reports/baseline"
)


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

        if (
            start == -1
            or end == -1
        ):
            return None


        return json.loads(
            text[start:end+1]
        )

    except Exception:

        return None



def set_match(
    pred,
    gt,
    key,
):

    return set(
        pred.get(key, [])
    ) == set(
        gt.get(key, [])
    )



def evaluate_file(path):

    records = load_jsonl(
        path
    )


    parse_success = 0
    schema_valid = 0

    primary_preds = []
    primary_labels = []

    exact_match = 0

    evidence_correct = 0
    recommendation_correct = 0


    for item in records:

        pred = parse_prediction(
            item["prediction"]
        )

        gt = item[
            "ground_truth"
        ]


        if pred is None:
            continue


        parse_success += 1


        required = [
            "task_type",
            "primary_issue",
            "severity",
            "evidence_codes",
            "recommended_action_codes",
            "explanation",
        ]


        if all(
            key in pred
            for key in required
        ):
            schema_valid += 1



        if (
            pred.get(
                "primary_issue"
            )
            ==
            gt.get(
                "primary_issue"
            )
        ):

            primary_preds.append(
                pred["primary_issue"]
            )

            primary_labels.append(
                gt["primary_issue"]
            )


        if (
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
            set_match(
                pred,
                gt,
                "evidence_codes"
            )
            and
            set_match(
                pred,
                gt,
                "recommended_action_codes"
            )
        ):

            exact_match += 1



        if set_match(
            pred,
            gt,
            "evidence_codes"
        ):
            evidence_correct += 1


        if set_match(
            pred,
            gt,
            "recommended_action_codes"
        ):
            recommendation_correct += 1



    total = len(records)


    result = {

        "total_samples":
            total,

        "parse_success_rate":
            parse_success / total,

        "schema_valid_rate":
            schema_valid / total,

        "core_exact_match_rate":
            exact_match / total,

        "primary_issue_accuracy":
            (
                accuracy_score(
                    primary_labels,
                    primary_preds,
                )
                if primary_labels
                else 0
            ),

        "primary_issue_macro_f1":
            (
                f1_score(
                    primary_labels,
                    primary_preds,
                    average="macro",
                    zero_division=0,
                )
                if primary_labels
                else 0
            ),

        "evidence_exact_set_accuracy":
            evidence_correct / total,

        "recommendation_exact_set_accuracy":
            recommendation_correct / total,
    }


    return result



def main():

    outputs = {}


    for name in [
        "standard_test",
        "hard_test",
    ]:

        path = (
            BASELINE_DIR
            /
            f"{name}_predictions.jsonl"
        )


        outputs[name] = evaluate_file(
            path
        )


    output_path = (
        OUTPUT_DIR
        /
        "baseline_metrics.json"
    )


    output_path.write_text(
        json.dumps(
            outputs,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "BASELINE EVALUATION PASSED"
    )


    print(
        json.dumps(
            outputs,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
