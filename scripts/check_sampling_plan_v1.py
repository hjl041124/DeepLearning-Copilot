import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path.cwd()

PLAN_PATH = ROOT / "configs" / "sampling_plan_v1.json"
TEMPLATE_PATH = ROOT / "configs" / "template_families_v1.json"


EXPECTED_SPLIT_TARGETS = {
    "train": 2400,
    "validation": 300,
    "test": 480,
}


EXPECTED_TASK_RATIOS = {
    "experiment_diagnosis": 0.60,
    "metric_interpretation": 0.20,
    "model_comparison": 0.20,
}


EXPECTED_PILOT_TOTAL = 144
EXPECTED_FULL_TOTAL = 3180


EXPECTED_DIAGNOSIS_ISSUES = {
    "overfitting",
    "underfitting",
    "optimization_problem",
    "class_imbalance",
    "data_quality_issue",
    "no_clear_issue",
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"缺少配置文件：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def check_balanced(values):
    if not values:
        return False

    return (
        max(values)
        - min(values)
        <= 1
    )


def main():
    errors = []

    try:
        plan = load_json(
            PLAN_PATH
        )

        templates = load_json(
            TEMPLATE_PATH
        )

    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        print(
            "DAY3 SAMPLING PLAN VALIDATION FAILED"
        )
        print("-", exc)
        raise SystemExit(1)

    if plan.get("version") != "1.0":
        errors.append(
            "sampling_plan_v1.json "
            "version 必须为 1.0"
        )

    if (
        plan.get(
            "final_split_targets"
        )
        != EXPECTED_SPLIT_TARGETS
    ):
        errors.append(
            "Split Targets（数据划分目标）"
            "与预期不一致"
        )

    if (
        plan.get(
            "task_type_ratios"
        )
        != EXPECTED_TASK_RATIOS
    ):
        errors.append(
            "Task Type Ratios（任务类型比例）"
            "与预期不一致"
        )

    policy = plan.get(
        "policy",
        {}
    )

    required_true_policies = [
        "balance_task_types_by_configured_ratio",
        "balance_diagnosis_primary_issues",
        "balance_scenarios_within_group",
        "balance_templates_within_scenario_and_split",
        "template_family_split_is_fixed",
        "pilot_is_for_pipeline_validation_only",
    ]

    for key in required_true_policies:
        if policy.get(key) is not True:
            errors.append(
                f"Sampling Policy（采样策略）"
                f"{key} 必须为 true"
            )

    if (
        policy.get(
            "random_split_after_generation"
        )
        is not False
    ):
        errors.append(
            "禁止 Dataset（数据集）生成后 "
            "执行 Random Split（随机划分）"
        )

    template_records = plan.get(
        "template_quotas",
        []
    )

    source_templates = templates[
        "template_families"
    ]

    source_template_map = {
        item["template_family_id"]: item
        for item in source_templates
    }

    quota_template_ids = [
        item["template_family_id"]
        for item in template_records
    ]

    if (
        len(quota_template_ids)
        != len(
            set(quota_template_ids)
        )
    ):
        errors.append(
            "发现重复 Template Family ID（模板族 ID）"
        )

    if (
        set(quota_template_ids)
        != set(source_template_map)
    ):
        errors.append(
            "Sampling Plan（采样计划）没有完整覆盖 "
            "Template Family（模板族）"
        )

    pilot_total = sum(
        item["pilot_sample_count"]
        for item in template_records
    )

    if (
        pilot_total
        != EXPECTED_PILOT_TOTAL
    ):
        errors.append(
            "Pilot Set（小规模试生成集）"
            f"应为 {EXPECTED_PILOT_TOTAL}，"
            f"当前为 {pilot_total}"
        )

    for item in template_records:
        if (
            item["pilot_sample_count"]
            != 1
        ):
            errors.append(
                f"{item['template_family_id']} "
                "Pilot Sample Count（试生成样本数）"
                "必须为 1"
            )

    full_total = sum(
        item["full_sample_count"]
        for item in template_records
    )

    if (
        full_total
        != EXPECTED_FULL_TOTAL
    ):
        errors.append(
            "Full Standard Set（正式标准集）"
            f"应为 {EXPECTED_FULL_TOTAL}，"
            f"当前为 {full_total}"
        )

    # --------------------------------------------------------
    # Split（数据划分）检查
    # --------------------------------------------------------

    split_counter = Counter()

    split_task_counter = defaultdict(
        Counter
    )

    split_issue_counter = defaultdict(
        Counter
    )

    scenario_split_counts = defaultdict(
        list
    )

    for item in template_records:
        template_id = item[
            "template_family_id"
        ]

        source_template = (
            source_template_map[
                template_id
            ]
        )

        if (
            item["split"]
            != source_template["split"]
        ):
            errors.append(
                f"{template_id} 的 Split（数据划分）"
                "与 Template Config（模板配置）不一致"
            )

        count = item[
            "full_sample_count"
        ]

        if count <= 0:
            errors.append(
                f"{template_id} 的正式生成数量 "
                "必须大于 0"
            )

        split_name = item[
            "split"
        ]

        task_type = item[
            "task_type"
        ]

        primary_issue = item[
            "primary_issue"
        ]

        scenario_id = item[
            "scenario_family_id"
        ]

        split_counter[
            split_name
        ] += count

        split_task_counter[
            split_name
        ][
            task_type
        ] += count

        if (
            task_type
            == "experiment_diagnosis"
        ):
            split_issue_counter[
                split_name
            ][
                primary_issue
            ] += count

        scenario_split_counts[
            (
                scenario_id,
                split_name
            )
        ].append(
            count
        )

    if (
        dict(split_counter)
        != EXPECTED_SPLIT_TARGETS
    ):
        errors.append(
            "Full Standard Set（正式标准集）的 "
            "Split Count（划分数量）错误："
            f"{dict(split_counter)}"
        )

    # --------------------------------------------------------
    # Task Type（任务类型）比例检查
    # --------------------------------------------------------

    for (
        split_name,
        split_total,
    ) in EXPECTED_SPLIT_TARGETS.items():

        expected_task_counts = {
            "experiment_diagnosis":
                round(
                    split_total
                    * 0.60
                ),
            "metric_interpretation":
                round(
                    split_total
                    * 0.20
                ),
        }

        expected_task_counts[
            "model_comparison"
        ] = (
            split_total
            - expected_task_counts[
                "experiment_diagnosis"
            ]
            - expected_task_counts[
                "metric_interpretation"
            ]
        )

        actual_task_counts = dict(
            split_task_counter[
                split_name
            ]
        )

        if (
            actual_task_counts
            != expected_task_counts
        ):
            errors.append(
                f"{split_name} 的 "
                "Task Type Distribution（任务类型分布）"
                "错误："
                f"{actual_task_counts}"
            )

    # --------------------------------------------------------
    # Diagnosis Primary Issue（诊断主要问题）平衡检查
    # --------------------------------------------------------

    for split_name in (
        EXPECTED_SPLIT_TARGETS
    ):
        issue_counts = (
            split_issue_counter[
                split_name
            ]
        )

        if (
            set(issue_counts)
            != EXPECTED_DIAGNOSIS_ISSUES
        ):
            errors.append(
                f"{split_name} 没有覆盖全部 "
                "Diagnosis Primary Issue（诊断主要问题）"
            )

        if not check_balanced(
            list(
                issue_counts.values()
            )
        ):
            errors.append(
                f"{split_name} 的 "
                "Diagnosis Primary Issue（诊断主要问题）"
                "分布不平衡："
                f"{dict(issue_counts)}"
            )

    # --------------------------------------------------------
    # Template Family（模板族）在 Scenario 内尽量平衡
    # --------------------------------------------------------

    for (
        scenario_split,
        values,
    ) in scenario_split_counts.items():
        if not check_balanced(
            values
        ):
            errors.append(
                "同一个 Scenario Family（场景族）"
                "内部的 Template Family（模板族）"
                "分配差异超过 1："
                f"{scenario_split} -> {values}"
            )

    if errors:
        print(
            "DAY3 SAMPLING PLAN VALIDATION FAILED"
        )

        for error in errors:
            print("-", error)

        raise SystemExit(1)

    print(
        "DAY3 SAMPLING PLAN VALIDATION PASSED"
    )

    print(
        "Pilot Set（小规模试生成集）：",
        pilot_total,
    )

    print(
        "Full Standard Set（正式标准集）：",
        full_total,
    )

    print(
        "Split Distribution（数据划分分布）：",
        dict(split_counter),
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        print(
            f"{split_name} Task Type Distribution"
            "（任务类型分布）：",
            dict(
                split_task_counter[
                    split_name
                ]
            ),
        )

        print(
            f"{split_name} Diagnosis Issue Distribution"
            "（诊断问题分布）：",
            dict(
                split_issue_counter[
                    split_name
                ]
            ),
        )


if __name__ == "__main__":
    main()
