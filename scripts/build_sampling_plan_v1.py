import json
from collections import defaultdict
from pathlib import Path


ROOT = Path.cwd()

SCENARIO_PATH = ROOT / "configs" / "scenario_families_v1.json"
TEMPLATE_PATH = ROOT / "configs" / "template_families_v1.json"
OUTPUT_PATH = ROOT / "configs" / "sampling_plan_v1.json"


FINAL_SPLIT_TARGETS = {
    "train": 2400,
    "validation": 300,
    "test": 480,
}


TASK_TYPE_RATIOS = {
    "experiment_diagnosis": 0.60,
    "metric_interpretation": 0.20,
    "model_comparison": 0.20,
}


PILOT_SAMPLES_PER_TEMPLATE = 1


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def balanced_allocate(total, keys):
    """
    将 total 尽可能平均地分配给 keys。
    最大数量与最小数量之差不会超过 1。
    """
    keys = sorted(keys)

    if not keys:
        raise ValueError("balanced_allocate received empty keys")

    base = total // len(keys)
    remainder = total % len(keys)

    result = {}

    for index, key in enumerate(keys):
        result[key] = base + (
            1 if index < remainder else 0
        )

    return result


def main():
    scenario_config = load_json(
        SCENARIO_PATH
    )

    template_config = load_json(
        TEMPLATE_PATH
    )

    templates = template_config[
        "template_families"
    ]

    # --------------------------------------------------------
    # Active Scenario Family（活动场景族）
    # --------------------------------------------------------

    active_scenarios = {}

    diagnosis_issue_to_scenarios = defaultdict(list)

    task_to_scenarios = defaultdict(list)

    for task_type, task_config in (
        scenario_config["task_types"].items()
    ):
        for scenario in task_config[
            "scenario_families"
        ]:
            if scenario.get("status") != "ready":
                continue

            if (
                scenario.get("standard_set_allowed")
                is not True
            ):
                continue

            scenario_id = scenario[
                "scenario_family_id"
            ]

            primary_issue = scenario[
                "primary_issue"
            ]

            active_scenarios[scenario_id] = {
                "task_type": task_type,
                "primary_issue": primary_issue,
            }

            task_to_scenarios[
                task_type
            ].append(
                scenario_id
            )

            if (
                task_type
                == "experiment_diagnosis"
            ):
                diagnosis_issue_to_scenarios[
                    primary_issue
                ].append(
                    scenario_id
                )

    # --------------------------------------------------------
    # Template Family（模板族）索引
    # --------------------------------------------------------

    template_by_id = {}

    templates_by_scenario_split = defaultdict(list)

    for template in templates:
        template_id = template[
            "template_family_id"
        ]

        scenario_id = template[
            "scenario_family_id"
        ]

        split_name = template[
            "split"
        ]

        template_by_id[
            template_id
        ] = template

        templates_by_scenario_split[
            (scenario_id, split_name)
        ].append(
            template_id
        )

    # --------------------------------------------------------
    # Pilot Plan（小规模试生成计划）
    # --------------------------------------------------------

    pilot_template_quotas = {}

    for template_id in sorted(
        template_by_id
    ):
        pilot_template_quotas[
            template_id
        ] = PILOT_SAMPLES_PER_TEMPLATE

    # --------------------------------------------------------
    # Full Generation Plan（正式生成计划）
    # --------------------------------------------------------

    full_template_quotas = {}

    split_task_targets = {}
    split_issue_targets = {}

    for split_name, split_total in (
        FINAL_SPLIT_TARGETS.items()
    ):
        # Task Type（任务类型）分配
        task_targets = {}

        running_total = 0

        task_names = list(
            TASK_TYPE_RATIOS.keys()
        )

        for index, task_type in enumerate(
            task_names
        ):
            if index < len(task_names) - 1:
                task_count = round(
                    split_total
                    * TASK_TYPE_RATIOS[
                        task_type
                    ]
                )

                running_total += task_count

            else:
                task_count = (
                    split_total
                    - running_total
                )

            task_targets[
                task_type
            ] = task_count

        split_task_targets[
            split_name
        ] = task_targets

        # ----------------------------------------------------
        # experiment_diagnosis（实验诊断）
        # ----------------------------------------------------

        diagnosis_total = task_targets[
            "experiment_diagnosis"
        ]

        diagnosis_issues = sorted(
            diagnosis_issue_to_scenarios
        )

        issue_targets = balanced_allocate(
            diagnosis_total,
            diagnosis_issues,
        )

        split_issue_targets[
            split_name
        ] = issue_targets

        for primary_issue in (
            diagnosis_issues
        ):
            scenario_ids = (
                diagnosis_issue_to_scenarios[
                    primary_issue
                ]
            )

            scenario_targets = (
                balanced_allocate(
                    issue_targets[
                        primary_issue
                    ],
                    scenario_ids,
                )
            )

            for (
                scenario_id,
                scenario_total,
            ) in scenario_targets.items():
                template_ids = (
                    templates_by_scenario_split[
                        (
                            scenario_id,
                            split_name,
                        )
                    ]
                )

                template_targets = (
                    balanced_allocate(
                        scenario_total,
                        template_ids,
                    )
                )

                full_template_quotas.update(
                    template_targets
                )

        # ----------------------------------------------------
        # metric_interpretation（指标解读）
        # model_comparison（模型比较）
        # ----------------------------------------------------

        for task_type in [
            "metric_interpretation",
            "model_comparison",
        ]:
            scenario_ids = (
                task_to_scenarios[
                    task_type
                ]
            )

            scenario_targets = (
                balanced_allocate(
                    task_targets[
                        task_type
                    ],
                    scenario_ids,
                )
            )

            for (
                scenario_id,
                scenario_total,
            ) in scenario_targets.items():
                template_ids = (
                    templates_by_scenario_split[
                        (
                            scenario_id,
                            split_name,
                        )
                    ]
                )

                template_targets = (
                    balanced_allocate(
                        scenario_total,
                        template_ids,
                    )
                )

                full_template_quotas.update(
                    template_targets
                )

    # --------------------------------------------------------
    # 构建详细 Template Quota（模板配额）
    # --------------------------------------------------------

    template_quota_records = []

    for template_id in sorted(
        template_by_id
    ):
        template = template_by_id[
            template_id
        ]

        template_quota_records.append(
            {
                "template_family_id":
                    template_id,
                "task_type":
                    template["task_type"],
                "primary_issue":
                    template["primary_issue"],
                "scenario_family_id":
                    template[
                        "scenario_family_id"
                    ],
                "presentation_family_id":
                    template[
                        "presentation_family_id"
                    ],
                "split":
                    template["split"],
                "pilot_sample_count":
                    pilot_template_quotas[
                        template_id
                    ],
                "full_sample_count":
                    full_template_quotas[
                        template_id
                    ],
            }
        )

    output = {
        "version": "1.0",
        "description": (
            "Sampling Plan for DeepLearning-Copilot "
            "Pilot Set and full Standard Set."
        ),
        "policy": {
            "sampling_order": [
                "split",
                "task_type",
                "primary_issue_for_experiment_diagnosis",
                "scenario_family",
                "template_family",
                "numeric_or_paraphrase_variant"
            ],
            "balance_task_types_by_configured_ratio":
                True,
            "balance_diagnosis_primary_issues":
                True,
            "balance_scenarios_within_group":
                True,
            "balance_templates_within_scenario_and_split":
                True,
            "random_split_after_generation":
                False,
            "template_family_split_is_fixed":
                True,
            "pilot_is_for_pipeline_validation_only":
                True
        },
        "final_split_targets":
            FINAL_SPLIT_TARGETS,
        "task_type_ratios":
            TASK_TYPE_RATIOS,
        "split_task_targets":
            split_task_targets,
        "split_diagnosis_issue_targets":
            split_issue_targets,
        "pilot": {
            "samples_per_template_family":
                PILOT_SAMPLES_PER_TEMPLATE,
            "expected_template_families":
                len(template_by_id),
            "expected_total_samples":
                (
                    len(template_by_id)
                    * PILOT_SAMPLES_PER_TEMPLATE
                )
        },
        "template_quotas":
            template_quota_records
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "SAMPLING PLAN BUILD PASSED"
    )

    print(
        "输出文件：",
        OUTPUT_PATH,
    )

    print(
        "Pilot Set（小规模试生成集）总数：",
        sum(
            record["pilot_sample_count"]
            for record
            in template_quota_records
        ),
    )

    print(
        "Full Standard Set（正式标准集）总数：",
        sum(
            record["full_sample_count"]
            for record
            in template_quota_records
        ),
    )

    print(
        "Split Targets（数据划分目标）：",
        FINAL_SPLIT_TARGETS,
    )

    print(
        "Task Type Ratios（任务类型比例）：",
        TASK_TYPE_RATIOS,
    )


if __name__ == "__main__":
    main()
