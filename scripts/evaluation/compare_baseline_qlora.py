import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


BASE_DIR = Path(
    "reports/baseline"
)

QLORA_DIR = Path(
    "reports/qlora"
)


OUTPUT = Path(
    "reports/baseline_vs_qlora_metrics.json"
)


ALLOWED_TASK_TYPES = {
    "experiment_diagnosis",
    "metric_interpretation",
    "model_comparison",
}


ALLOWED_ISSUES = {
    "overfitting",
    "underfitting",
    "optimization_problem",
    "class_imbalance",
    "data_quality_issue",
    "no_clear_issue",
    "not_applicable",
}


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



def evaluate(path):

    records = load_jsonl(
        path
    )

    total = len(records)

    parse_success = 0
    task_valid = 0
    issue_valid = 0

    issue_pred = []
    issue_true = []

    evidence_match = 0
    action_match = 0
    core_match = 0


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


        task_ok = (
            pred.get("task_type")
            in ALLOWED_TASK_TYPES
        )

        issue_ok = (
            pred.get("primary_issue")
            in ALLOWED_ISSUES
        )


        if task_ok:
            task_valid += 1


        if issue_ok:
            issue_valid += 1


            issue_pred.append(
                pred["primary_issue"]
            )

            issue_true.append(
                gt["primary_issue"]
            )


        evidence_ok = (
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
        )


        action_ok = (
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


        if evidence_ok:
            evidence_match += 1


        if action_ok:
            action_match += 1


        if (
            task_ok
            and issue_ok
            and pred.get("severity")
            ==
            gt.get("severity")
            and evidence_ok
            and action_ok
        ):
            core_match += 1



    return {

        "samples":
            total,

        "parse_success_rate":
            parse_success / total,

        "task_type_valid_rate":
            task_valid / total,

        "primary_issue_valid_rate":
            issue_valid / total,

        "primary_issue_accuracy":
            (
                accuracy_score(
                    issue_true,
                    issue_pred,
                )
                if issue_true
                else 0
            ),

        "primary_issue_macro_f1":
            (
                f1_score(
                    issue_true,
                    issue_pred,
                    average="macro",
                    zero_division=0,
                )
                if issue_true
                else 0
            ),

        "evidence_exact_match":
            evidence_match / total,

        "recommendation_exact_match":
            action_match / total,

        "core_exact_match":
            core_match / total,
    }



def main():

    result = {}

    for dataset in [
        "standard_test",
        "hard_test",
    ]:

        result[dataset] = {

            "baseline":
                evaluate(
                    BASE_DIR
                    /
                    f"{dataset}_predictions.jsonl"
                ),

            "qlora":
                evaluate(
                    QLORA_DIR
                    /
                    f"{dataset}_predictions.jsonl"
                ),
        }


    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "BASELINE VS QLORA EVALUATION PASSED"
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
