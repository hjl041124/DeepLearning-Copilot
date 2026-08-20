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


OUTPUT_JSON = (
    BASELINE_DIR
    /
    "baseline_metrics_v2.json"
)


OUTPUT_MD = (
    BASELINE_DIR
    /
    "baseline_summary.md"
)


ALLOWED_TASK_TYPES = {
    "experiment_diagnosis",
    "metric_interpretation",
    "model_comparison",
}


ALLOWED_PRIMARY_ISSUES = {
    "overfitting",
    "underfitting",
    "optimization_problem",
    "class_imbalance",
    "data_quality_issue",
    "no_clear_issue",
    "not_applicable",
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

        if start == -1 or end == -1:
            return None

        return json.loads(
            text[start:end+1]
        )

    except Exception:

        return None



def evaluate_dataset(path):

    records = load_jsonl(path)

    total = len(records)

    parse_success = 0
    task_valid = 0
    issue_valid = 0

    core_success = 0

    issue_pred = []
    issue_true = []

    evidence_exact = 0
    action_exact = 0


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
            pred.get(
                "task_type"
            )
            in ALLOWED_TASK_TYPES
        )


        issue_ok = (
            pred.get(
                "primary_issue"
            )
            in ALLOWED_PRIMARY_ISSUES
        )


        if task_ok:
            task_valid += 1


        if issue_ok:
            issue_valid += 1


        if issue_ok:

            issue_pred.append(
                pred[
                    "primary_issue"
                ]
            )

            issue_true.append(
                gt[
                    "primary_issue"
                ]
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
            evidence_exact += 1


        if action_ok:
            action_exact += 1



        if (
            task_ok
            and
            issue_ok
            and
            pred.get(
                "severity"
            )
            ==
            gt.get(
                "severity"
            )
            and evidence_ok
            and action_ok
        ):
            core_success += 1



    result = {

        "total_samples":
            total,

        "semantic_metrics": {

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
        },


        "protocol_metrics": {

            "parse_success_rate":
                parse_success / total,

            "task_type_valid_rate":
                task_valid / total,

            "primary_issue_valid_rate":
                issue_valid / total,
        },


        "grounding_metrics": {

            "evidence_exact_set_accuracy":
                evidence_exact / total,

            "recommendation_exact_set_accuracy":
                action_exact / total,
        },


        "complete_success_metrics": {

            "core_exact_match_rate":
                core_success / total,
        }
    }


    return result



def main():

    results = {}


    for name in [
        "standard_test",
        "hard_test",
    ]:

        results[name] = evaluate_dataset(
            BASELINE_DIR
            /
            f"{name}_predictions.jsonl"
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
        "# Baseline Evaluation Report\n"
    )

    md.append(
        "Model: Qwen3-4B-Instruct-2507 Base Model\n"
    )


    for name, result in results.items():

        md.append(
            f"\n## {name}\n"
        )

        for group, values in result.items():

            md.append(
                f"\n### {group}\n"
            )

            if isinstance(values, dict):

                for key, value in values.items():

                    md.append(
                        f"- {key}: {value:.4f}\n"
                    )

            else:

                md.append(
                    f"- value: {values:.4f}\n"
                )


    OUTPUT_MD.write_text(
        "".join(md),
        encoding="utf-8",
    )


    print(
        "BASELINE EVALUATION V2 PASSED"
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
