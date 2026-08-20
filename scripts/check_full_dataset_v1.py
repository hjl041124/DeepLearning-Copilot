import json
from collections import Counter
from pathlib import Path

from scripts.check_pilot_dataset_v1 import (
    canonical,
    extract_diagnosis_ground_truth,
)

from src.evaluation.ground_truth_builder import (
    build_ground_truth,
)

from src.evaluation.non_diagnosis_ground_truth import (
    build_non_diagnosis_ground_truth,
)

from src.evaluation.output_validator import (
    validate_output,
)

from src.evaluation.dataset_text_renderer import (
    enrich_ground_truth_explanation,
)


ROOT = Path.cwd()

DATA_PATH = (
    ROOT
    / "data"
    / "generated"
    / "full_dataset_v1.jsonl"
)

TEMPLATE_PATH = (
    ROOT
    / "configs"
    / "template_families_v1.json"
)

SAMPLING_PATH = (
    ROOT
    / "configs"
    / "sampling_plan_v1.json"
)


EXPECTED_TOTAL = 3180

EXPECTED_SPLITS = {
    "train": 2400,
    "validation": 300,
    "test": 480,
}

EXPECTED_TASKS = {
    "experiment_diagnosis": 1908,
    "metric_interpretation": 636,
    "model_comparison": 636,
}

EXPECTED_DIAGNOSIS_ISSUES = {
    "overfitting": 318,
    "underfitting": 318,
    "optimization_problem": 318,
    "class_imbalance": 318,
    "data_quality_issue": 318,
    "no_clear_issue": 318,
}


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL line "
                    f"{line_number}: {exc}"
                )

    return records


def recompute_ground_truth(
    sample,
):
    raw_record = sample[
        "raw_record"
    ]

    task_type = sample[
        "task_type"
    ]

    scenario_id = sample[
        "scenario_family_id"
    ]

    if (
        task_type
        == "experiment_diagnosis"
    ):
        base = (
            extract_diagnosis_ground_truth(
                build_ground_truth(
                    raw_record
                )
            )
        )

    else:
        base = (
            build_non_diagnosis_ground_truth(
                raw_record
            )
        )

    return (
        enrich_ground_truth_explanation(
            task_type=task_type,
            scenario_id=scenario_id,
            raw_record=raw_record,
            ground_truth=base,
        )
    )


def main():
    errors = []

    try:
        samples = load_jsonl(
            DATA_PATH
        )

        template_config = load_json(
            TEMPLATE_PATH
        )

        sampling_plan = load_json(
            SAMPLING_PATH
        )

    except Exception as exc:
        print(
            "DAY3 FULL DATASET VALIDATION FAILED"
        )
        print("-", exc)
        raise SystemExit(1)

    if len(samples) != EXPECTED_TOTAL:
        errors.append(
            f"Full Dataset（正式数据集）"
            f"应为 {EXPECTED_TOTAL} 条，"
            f"当前为 {len(samples)}"
        )

    template_map = {
        item[
            "template_family_id"
        ]: item
        for item
        in template_config[
            "template_families"
        ]
    }

    quota_map = {
        item[
            "template_family_id"
        ]:
            item[
                "full_sample_count"
            ]
        for item
        in sampling_plan[
            "template_quotas"
        ]
    }

    sample_ids = [
        item[
            "sample_id"
        ]
        for item in samples
    ]

    if (
        len(sample_ids)
        != len(set(sample_ids))
    ):
        errors.append(
            "发现重复 Sample ID（样本 ID）"
        )

    prompts = [
        item[
            "prompt"
        ].strip()
        for item in samples
    ]

    if (
        len(prompts)
        != len(set(prompts))
    ):
        errors.append(
            "发现完全重复 Prompt（提示词）"
        )

    template_counter = Counter()
    split_counter = Counter()
    task_counter = Counter()
    issue_counter = Counter()
    severity_counter = Counter()

    recompute_failures = 0

    for sample in samples:
        sample_id = sample[
            "sample_id"
        ]

        template_id = sample[
            "template_family_id"
        ]

        scenario_id = sample[
            "scenario_family_id"
        ]

        if (
            template_id
            not in template_map
        ):
            errors.append(
                f"{sample_id}: 未知 "
                "Template Family（模板族）"
            )
            continue

        template = template_map[
            template_id
        ]

        if (
            sample["split"]
            != template["split"]
        ):
            errors.append(
                f"{sample_id}: Split（数据划分）"
                "与 Template Family（模板族）不一致"
            )

        if (
            sample[
                "scenario_family_id"
            ]
            != template[
                "scenario_family_id"
            ]
        ):
            errors.append(
                f"{sample_id}: "
                "Scenario Family（场景族）"
                "与 Template Family（模板族）不一致"
            )

        if (
            sample[
                "task_type"
            ]
            != template[
                "task_type"
            ]
        ):
            errors.append(
                f"{sample_id}: "
                "Task Type（任务类型）"
                "与 Template Family（模板族）不一致"
            )

        raw_record = sample[
            "raw_record"
        ]

        if (
            "ground_truth"
            in raw_record
        ):
            errors.append(
                f"{sample_id}: "
                "Raw Record（原始记录）"
                "不得包含 ground_truth"
            )

        if (
            "primary_issue"
            in raw_record
        ):
            errors.append(
                f"{sample_id}: "
                "Raw Record（原始记录）"
                "不得包含 primary_issue"
            )

        output_errors = (
            validate_output(
                sample[
                    "ground_truth"
                ]
            )
        )

        if output_errors:
            errors.append(
                f"{sample_id}: "
                "Output Schema（输出结构）"
                "验证失败："
                + "; ".join(
                    output_errors
                )
            )

        try:
            recomputed = (
                recompute_ground_truth(
                    sample
                )
            )

        except Exception as exc:
            recompute_failures += 1

            errors.append(
                f"{sample_id}: "
                "Ground Truth Recompute"
                f"（标准答案重计算）失败：{exc}"
            )

            continue

        if (
            canonical(
                recomputed
            )
            != canonical(
                sample[
                    "ground_truth"
                ]
            )
        ):
            errors.append(
                f"{sample_id}: "
                "Ground Truth（标准答案）"
                "重计算结果不一致"
            )

        messages = sample.get(
            "messages",
            []
        )

        if len(messages) != 3:
            errors.append(
                f"{sample_id}: "
                "Messages（对话格式）"
                "必须有 3 条"
            )

        else:
            roles = [
                item.get(
                    "role"
                )
                for item
                in messages
            ]

            if roles != [
                "system",
                "user",
                "assistant",
            ]:
                errors.append(
                    f"{sample_id}: "
                    "Message Role（对话角色）"
                    "顺序错误"
                )

            if (
                messages[1].get(
                    "content"
                )
                != sample[
                    "prompt"
                ]
            ):
                errors.append(
                    f"{sample_id}: "
                    "User Message（用户消息）"
                    "与 Prompt（提示词）不一致"
                )

            try:
                assistant_target = (
                    json.loads(
                        messages[
                            2
                        ][
                            "content"
                        ]
                    )
                )

                if (
                    canonical(
                        assistant_target
                    )
                    != canonical(
                        sample[
                            "ground_truth"
                        ]
                    )
                ):
                    errors.append(
                        f"{sample_id}: "
                        "Assistant Target（助手目标答案）"
                        "与 Ground Truth（标准答案）不一致"
                    )

            except json.JSONDecodeError:
                errors.append(
                    f"{sample_id}: "
                    "Assistant Target（助手目标答案）"
                    "不是合法 JSON"
                )

        prompt = sample[
            "prompt"
        ]

        if (
            scenario_id
            in prompt
        ):
            errors.append(
                f"{sample_id}: "
                "Prompt（提示词）泄漏 "
                "Scenario Family ID（场景族 ID）"
            )

        if (
            template_id
            in prompt
        ):
            errors.append(
                f"{sample_id}: "
                "Prompt（提示词）泄漏 "
                "Template Family ID（模板族 ID）"
            )

        if (
            sample[
                "task_type"
            ]
            == "experiment_diagnosis"
            and sample[
                "primary_issue"
            ]
            in prompt
        ):
            errors.append(
                f"{sample_id}: "
                "Prompt（提示词）直接泄漏 "
                "Primary Issue（主要问题）"
            )

        template_counter[
            template_id
        ] += 1

        split_counter[
            sample[
                "split"
            ]
        ] += 1

        task_counter[
            sample[
                "task_type"
            ]
        ] += 1

        if (
            sample[
                "task_type"
            ]
            == "experiment_diagnosis"
        ):
            issue_counter[
                sample[
                    "primary_issue"
                ]
            ] += 1

        severity_counter[
            sample[
                "ground_truth"
            ][
                "severity"
            ]
        ] += 1

    for (
        template_id,
        expected_count,
    ) in quota_map.items():

        actual_count = (
            template_counter[
                template_id
            ]
        )

        if (
            actual_count
            != expected_count
        ):
            errors.append(
                f"{template_id}: "
                "Template Quota（模板配额）"
                f"应为 {expected_count}，"
                f"当前为 {actual_count}"
            )

    if (
        dict(split_counter)
        != EXPECTED_SPLITS
    ):
        errors.append(
            "Split Distribution（数据划分分布）"
            f"错误：{dict(split_counter)}"
        )

    if (
        dict(task_counter)
        != EXPECTED_TASKS
    ):
        errors.append(
            "Task Type Distribution（任务类型分布）"
            f"错误：{dict(task_counter)}"
        )

    if (
        dict(issue_counter)
        != EXPECTED_DIAGNOSIS_ISSUES
    ):
        errors.append(
            "Diagnosis Issue Distribution"
            "（诊断主要问题分布）错误："
            f"{dict(issue_counter)}"
        )

    split_paths = {
        "train":
            ROOT
            / "data"
            / "generated"
            / "full_train_v1.jsonl",

        "validation":
            ROOT
            / "data"
            / "generated"
            / "full_validation_v1.jsonl",

        "test":
            ROOT
            / "data"
            / "generated"
            / "full_test_v1.jsonl",
    }

    master_ids_by_split = {}

    for split_name in (
        EXPECTED_SPLITS
    ):
        master_ids_by_split[
            split_name
        ] = {
            sample[
                "sample_id"
            ]
            for sample in samples
            if sample[
                "split"
            ]
            == split_name
        }

    for (
        split_name,
        split_path,
    ) in split_paths.items():

        if not split_path.exists():
            errors.append(
                f"缺少 Split JSONL（划分数据文件）："
                f"{split_path}"
            )
            continue

        split_samples = (
            load_jsonl(
                split_path
            )
        )

        actual_ids = {
            item[
                "sample_id"
            ]
            for item
            in split_samples
        }

        if (
            actual_ids
            != master_ids_by_split[
                split_name
            ]
        ):
            errors.append(
                f"{split_name} "
                "Split JSONL（划分数据文件）"
                "与 Master JSONL（主数据文件）不一致"
            )

    if errors:
        print(
            "DAY3 FULL DATASET VALIDATION FAILED"
        )

        print(
            "错误数量：",
            len(errors),
        )

        for error in errors[
            :50
        ]:
            print(
                "-",
                error,
            )

        if len(errors) > 50:
            print(
                "... additional errors:",
                len(errors) - 50,
            )

        raise SystemExit(1)

    print(
        "DAY3 FULL DATASET VALIDATION PASSED"
    )

    print(
        "Full Samples（正式样本）：",
        len(samples),
    )

    print(
        "Unique Sample IDs（唯一样本 ID）：",
        len(
            set(sample_ids)
        ),
    )

    print(
        "Unique Prompts（唯一提示词）：",
        len(
            set(prompts)
        ),
    )

    print(
        "Unique Template Families（唯一模板族）：",
        len(
            template_counter
        ),
    )

    print(
        "Split Distribution（数据划分分布）：",
        dict(split_counter),
    )

    print(
        "Task Type Distribution（任务类型分布）：",
        dict(task_counter),
    )

    print(
        "Diagnosis Issue Distribution"
        "（诊断主要问题分布）：",
        dict(issue_counter),
    )

    print(
        "Severity Distribution（严重程度分布）：",
        dict(severity_counter),
    )

    print(
        "Ground Truth Recompute Check"
        "（标准答案重计算检查）：PASSED"
    )

    print(
        "Template Leakage Check"
        "（模板泄漏检查）：PASSED"
    )

    print(
        "Duplicate Prompt Check"
        "（重复提示词检查）：PASSED"
    )


if __name__ == "__main__":
    main()
