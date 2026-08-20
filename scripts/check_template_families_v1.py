import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path.cwd()

SCENARIO_PATH = ROOT / "configs" / "scenario_families_v1.json"
TEMPLATE_PATH = ROOT / "configs" / "template_families_v1.json"


EXPECTED_PRESENTATION_SPLITS = {
    "PF_STRUCTURED_BLOCK": "train",
    "PF_TABULAR_REPORT": "train",
    "PF_TRACKER_EXPORT": "train",
    "PF_CONCISE_NOTE": "validation",
    "PF_DEBUG_TICKET": "test",
    "PF_NARRATIVE_SUMMARY": "test",
}

EXPECTED_PER_SCENARIO_SPLITS = {
    "train": 3,
    "validation": 1,
    "test": 2,
}

EXPECTED_ACTIVE_SCENARIOS = 24
EXPECTED_TEMPLATE_FAMILIES = 144

EXPECTED_TASK_TEMPLATE_COUNTS = {
    "experiment_diagnosis": 96,
    "metric_interpretation": 24,
    "model_comparison": 24,
}

EXPECTED_SPLIT_TEMPLATE_COUNTS = {
    "train": 72,
    "validation": 24,
    "test": 48,
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"缺少配置文件：{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    errors = []

    try:
        scenario_config = load_json(
            SCENARIO_PATH
        )
        template_config = load_json(
            TEMPLATE_PATH
        )

    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        print(
            "DAY3 TEMPLATE FAMILY VALIDATION FAILED"
        )
        print("-", exc)
        raise SystemExit(1)

    if template_config.get("version") != "1.0":
        errors.append(
            "template_families_v1.json "
            "的 version 必须为 1.0"
        )

    policy = template_config.get(
        "design_policy",
        {},
    )

    if (
        policy.get("standard_split_unit")
        != "template_family_id"
    ):
        errors.append(
            "standard_split_unit 必须为 "
            "template_family_id"
        )

    if (
        policy.get(
            "split_assignment_before_generation"
        )
        is not True
    ):
        errors.append(
            "split assignment 必须发生在数据生成前"
        )

    if (
        policy.get(
            "random_split_after_generation"
        )
        is not False
    ):
        errors.append(
            "禁止在生成 Dataset 后执行 random split"
        )

    if (
        policy.get(
            "scenario_family_may_span_standard_splits"
        )
        is not True
    ):
        errors.append(
            "scenario_family 必须允许通过不同 "
            "template_family 跨 Standard Split"
        )

    if (
        policy.get(
            "unseen_scenario_family_reserved_for_hard_test"
        )
        is not True
    ):
        errors.append(
            "完全 unseen scenario_family "
            "必须保留给 Hard Test"
        )

    presentation_families = (
        template_config.get(
            "presentation_families",
            [],
        )
    )

    presentation_map = {}

    for presentation in presentation_families:
        presentation_id = presentation.get(
            "presentation_family_id"
        )

        split_name = presentation.get(
            "split"
        )

        if not presentation_id:
            errors.append(
                "发现缺少 presentation_family_id "
                "的配置"
            )
            continue

        if presentation_id in presentation_map:
            errors.append(
                "发现重复 presentation_family_id："
                f"{presentation_id}"
            )

        presentation_map[
            presentation_id
        ] = split_name

    if (
        presentation_map
        != EXPECTED_PRESENTATION_SPLITS
    ):
        errors.append(
            "Presentation Family（呈现结构族）"
            "与预期 split 配置不一致："
            f"{presentation_map}"
        )

    active_scenarios = {}

    for task_type, task_config in (
        scenario_config.get(
            "task_types",
            {},
        ).items()
    ):
        for scenario in task_config.get(
            "scenario_families",
            [],
        ):
            if scenario.get("status") != "ready":
                continue

            if (
                scenario.get(
                    "standard_set_allowed"
                )
                is not True
            ):
                continue

            scenario_id = scenario[
                "scenario_family_id"
            ]

            active_scenarios[scenario_id] = {
                "task_type": task_type,
                "primary_issue":
                    scenario["primary_issue"],
                "required_inputs":
                    scenario.get(
                        "required_inputs",
                        [],
                    ),
            }

    if (
        len(active_scenarios)
        != EXPECTED_ACTIVE_SCENARIOS
    ):
        errors.append(
            "active scenario family 数量应为 "
            f"{EXPECTED_ACTIVE_SCENARIOS}，当前为 "
            f"{len(active_scenarios)}"
        )

    deferred_ids = {
        item["scenario_family_id"]
        for item in scenario_config.get(
            "deferred_scenario_families",
            [],
        )
    }

    template_families = (
        template_config.get(
            "template_families",
            [],
        )
    )

    if (
        len(template_families)
        != EXPECTED_TEMPLATE_FAMILIES
    ):
        errors.append(
            "Template Family（模板族）数量应为 "
            f"{EXPECTED_TEMPLATE_FAMILIES}，当前为 "
            f"{len(template_families)}"
        )

    template_ids = []
    task_counter = Counter()
    split_counter = Counter()

    scenario_templates = defaultdict(list)
    scenario_presentations = defaultdict(set)
    scenario_split_counter = defaultdict(Counter)

    for template in template_families:
        template_id = template.get(
            "template_family_id"
        )

        scenario_id = template.get(
            "scenario_family_id"
        )

        presentation_id = template.get(
            "presentation_family_id"
        )

        split_name = template.get(
            "split"
        )

        task_type = template.get(
            "task_type"
        )

        if not template_id:
            errors.append(
                "发现缺少 template_family_id 的配置"
            )
            continue

        template_ids.append(
            template_id
        )

        if scenario_id not in active_scenarios:
            errors.append(
                f"{template_id} 引用了非 active "
                f"scenario_family：{scenario_id}"
            )
            continue

        if scenario_id in deferred_ids:
            errors.append(
                f"{template_id} 错误引用 deferred "
                f"scenario_family：{scenario_id}"
            )

        expected_scenario = active_scenarios[
            scenario_id
        ]

        if (
            task_type
            != expected_scenario["task_type"]
        ):
            errors.append(
                f"{template_id} 的 task_type "
                "与 Scenario Family（场景族）不一致"
            )

        if (
            template.get("primary_issue")
            != expected_scenario["primary_issue"]
        ):
            errors.append(
                f"{template_id} 的 primary_issue "
                "与 Scenario Family（场景族）不一致"
            )

        expected_inputs = set(
            expected_scenario[
                "required_inputs"
            ]
        )

        actual_inputs = set(
            template.get(
                "required_inputs",
                [],
            )
        )

        if actual_inputs != expected_inputs:
            errors.append(
                f"{template_id} 的 required_inputs "
                "与 Scenario Family（场景族）不一致"
            )

        if (
            presentation_id
            not in presentation_map
        ):
            errors.append(
                f"{template_id} 引用了未知 "
                f"presentation_family_id："
                f"{presentation_id}"
            )
            continue

        expected_split = presentation_map[
            presentation_id
        ]

        if split_name != expected_split:
            errors.append(
                f"{template_id} 的 split "
                f"应为 {expected_split}，"
                f"当前为 {split_name}"
            )

        if template.get("status") != "ready":
            errors.append(
                f"{template_id} 必须标记为 ready"
            )

        variant_policy = template.get(
            "variant_policy",
            {},
        )

        if (
            variant_policy.get(
                "variants_must_inherit_split"
            )
            is not True
        ):
            errors.append(
                f"{template_id} 未要求 numeric/paraphrase "
                "variants 继承 split"
            )

        if (
            variant_policy.get(
                "surface_paraphrase_does_not_create_new_family"
            )
            is not True
        ):
            errors.append(
                f"{template_id} 未明确禁止使用表面 "
                "paraphrase 创建新 Template Family（模板族）"
            )

        if not template.get(
            "render_contract"
        ):
            errors.append(
                f"{template_id} 缺少 render_contract"
            )

        scenario_templates[
            scenario_id
        ].append(
            template_id
        )

        scenario_presentations[
            scenario_id
        ].add(
            presentation_id
        )

        scenario_split_counter[
            scenario_id
        ][
            split_name
        ] += 1

        task_counter[
            task_type
        ] += 1

        split_counter[
            split_name
        ] += 1

    duplicate_template_ids = sorted(
        template_id
        for template_id, count
        in Counter(template_ids).items()
        if count > 1
    )

    if duplicate_template_ids:
        errors.append(
            "发现重复 template_family_id："
            f"{duplicate_template_ids}"
        )

    for scenario_id in active_scenarios:
        template_count = len(
            scenario_templates[
                scenario_id
            ]
        )

        if template_count != 6:
            errors.append(
                f"{scenario_id} 应有 6 个 "
                "Template Family（模板族），当前为 "
                f"{template_count}"
            )

        presentations = (
            scenario_presentations[
                scenario_id
            ]
        )

        if presentations != set(
            EXPECTED_PRESENTATION_SPLITS
        ):
            errors.append(
                f"{scenario_id} 没有覆盖全部 6 个 "
                "Presentation Family（呈现结构族）"
            )

        actual_split_counts = dict(
            scenario_split_counter[
                scenario_id
            ]
        )

        if (
            actual_split_counts
            != EXPECTED_PER_SCENARIO_SPLITS
        ):
            errors.append(
                f"{scenario_id} 的 split family "
                "分布错误："
                f"{actual_split_counts}"
            )

    if (
        dict(task_counter)
        != EXPECTED_TASK_TEMPLATE_COUNTS
    ):
        errors.append(
            "Task Type（任务类型）的 Template Family "
            "数量不正确："
            f"{dict(task_counter)}"
        )

    if (
        dict(split_counter)
        != EXPECTED_SPLIT_TEMPLATE_COUNTS
    ):
        errors.append(
            "Split（数据划分）的 Template Family "
            "数量不正确："
            f"{dict(split_counter)}"
        )

    if errors:
        print(
            "DAY3 TEMPLATE FAMILY VALIDATION FAILED"
        )

        for error in errors:
            print("-", error)

        raise SystemExit(1)

    print(
        "DAY3 TEMPLATE FAMILY VALIDATION PASSED"
    )
    print(
        "配置文件：",
        TEMPLATE_PATH,
    )
    print(
        "Active Scenario Family（活动场景族）：",
        len(active_scenarios),
    )
    print(
        "Presentation Family（呈现结构族）：",
        len(presentation_families),
    )
    print(
        "Template Family（模板族）：",
        len(template_families),
    )
    print(
        "Task Type（任务类型）分布：",
        dict(task_counter),
    )
    print(
        "Split（数据划分）分布：",
        dict(split_counter),
    )
    print(
        "每个 Scenario Family（场景族）的 Split 配置：",
        EXPECTED_PER_SCENARIO_SPLITS,
    )


if __name__ == "__main__":
    main()
