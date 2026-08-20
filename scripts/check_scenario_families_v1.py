import json
from collections import Counter
from pathlib import Path


ROOT = Path.cwd()

SCENARIO_PATH = ROOT / "configs" / "scenario_families_v1.json"
TAXONOMY_PATH = ROOT / "configs" / "taxonomy_v1.json"
FEATURE_PATH = ROOT / "configs" / "feature_definitions_v1.json"
NON_DIAGNOSIS_BUILDER_PATH = (
    ROOT
    / "src"
    / "evaluation"
    / "non_diagnosis_ground_truth.py"
)


EXPECTED_TASK_TYPES = {
    "experiment_diagnosis",
    "metric_interpretation",
    "model_comparison",
}

EXPECTED_DIAGNOSIS_ISSUES = {
    "overfitting",
    "underfitting",
    "optimization_problem",
    "class_imbalance",
    "data_quality_issue",
    "no_clear_issue",
}

EXPECTED_ACTIVE_COUNTS = {
    "experiment_diagnosis": 16,
    "metric_interpretation": 4,
    "model_comparison": 4,
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"缺少配置文件：{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_name_set(value):
    names = set()

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                names.add(item)

            elif isinstance(item, dict):
                for key in (
                    "name",
                    "id",
                    "task_type",
                    "primary_issue",
                    "value",
                ):
                    if (
                        key in item
                        and isinstance(item[key], str)
                    ):
                        names.add(item[key])
                        break

    elif isinstance(value, dict):
        names.update(
            str(key)
            for key in value.keys()
        )

    return names


def main() -> None:
    errors = []

    try:
        scenario_config = load_json(
            SCENARIO_PATH
        )
        taxonomy = load_json(
            TAXONOMY_PATH
        )
        feature_config = load_json(
            FEATURE_PATH
        )

    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        print(
            "DAY3 SCENARIO FAMILY VALIDATION FAILED"
        )
        print("-", exc)
        raise SystemExit(1)

    if not NON_DIAGNOSIS_BUILDER_PATH.exists():
        errors.append(
            "缺少 non_diagnosis_ground_truth.py，"
            "metric_interpretation 和 model_comparison "
            "不能标记为 ready"
        )

    if scenario_config.get("version") != "1.0":
        errors.append(
            "scenario_families_v1.json "
            "的 version 必须为 1.0"
        )

    design_policy = scenario_config.get(
        "design_policy",
        {},
    )

    if (
        design_policy.get(
            "standard_split_unit"
        )
        != "template_family_id"
    ):
        errors.append(
            "standard_split_unit 必须为 "
            "template_family_id"
        )

    if (
        design_policy.get(
            "scenario_family_may_span_standard_splits"
        )
        is not True
    ):
        errors.append(
            "scenario_family_may_span_standard_splits "
            "必须为 true"
        )

    if (
        design_policy.get(
            "unseen_scenario_family_reserved_for_hard_test"
        )
        is not True
    ):
        errors.append(
            "unseen scenario_family "
            "必须保留给 Hard Test"
        )

    if (
        design_policy.get(
            "ground_truth_must_be_deterministic_python"
        )
        is not True
    ):
        errors.append(
            "Ground Truth 必须明确由 "
            "deterministic Python 生成"
        )

    task_configs = scenario_config.get(
        "task_types",
        {},
    )

    task_type_names = set(
        task_configs.keys()
    )

    if (
        task_type_names
        != EXPECTED_TASK_TYPES
    ):
        errors.append(
            "scenario config 的 task_types 不正确："
            f"{sorted(task_type_names)}"
        )

    taxonomy_task_types = extract_name_set(
        taxonomy.get(
            "task_types",
            [],
        )
    )

    if (
        taxonomy_task_types
        and not EXPECTED_TASK_TYPES.issubset(
            taxonomy_task_types
        )
    ):
        errors.append(
            "scenario config 与 taxonomy_v1.json "
            "的 task_types 不一致"
        )

    taxonomy_primary_issues = extract_name_set(
        taxonomy.get(
            "primary_issues",
            [],
        )
    )

    if taxonomy_primary_issues:
        missing_issues = (
            EXPECTED_DIAGNOSIS_ISSUES
            - taxonomy_primary_issues
        )

        if missing_issues:
            errors.append(
                "taxonomy_v1.json 缺少 "
                "diagnosis primary issues："
                f"{sorted(missing_issues)}"
            )

        if (
            "not_applicable"
            not in taxonomy_primary_issues
        ):
            errors.append(
                "taxonomy_v1.json "
                "缺少 not_applicable"
            )

    feature_text = json.dumps(
        feature_config,
        ensure_ascii=False,
        sort_keys=True,
    )

    all_ids = []
    active_counts = {}
    issue_counter = Counter()

    for task_type, task_config in (
        task_configs.items()
    ):
        scenarios = task_config.get(
            "scenario_families",
            [],
        )

        active_counts[task_type] = len(
            scenarios
        )

        expected_count = (
            EXPECTED_ACTIVE_COUNTS.get(
                task_type
            )
        )

        if (
            expected_count is not None
            and len(scenarios)
            != expected_count
        ):
            errors.append(
                f"{task_type} 应有 "
                f"{expected_count} 个 active "
                "scenario families，当前为 "
                f"{len(scenarios)}"
            )

        if (
            task_config.get("status")
            != "ready_for_template_design"
        ):
            errors.append(
                f"{task_type} 当前应标记为 "
                "ready_for_template_design"
            )

        for scenario in scenarios:
            scenario_id = scenario.get(
                "scenario_family_id"
            )

            primary_issue = scenario.get(
                "primary_issue"
            )

            status = scenario.get(
                "status"
            )

            if not scenario_id:
                errors.append(
                    f"{task_type} 中发现缺少 "
                    "scenario_family_id 的配置"
                )
                continue

            all_ids.append(
                scenario_id
            )

            if not scenario.get(
                "source_basis"
            ):
                errors.append(
                    f"{scenario_id} "
                    "缺少 source_basis"
                )

            if not scenario.get(
                "ground_truth_strategy"
            ):
                errors.append(
                    f"{scenario_id} "
                    "缺少 ground_truth_strategy"
                )

            required_features = (
                scenario.get(
                    "required_features",
                    [],
                )
            )

            for feature_name in (
                required_features
            ):
                if (
                    feature_name
                    not in feature_text
                ):
                    errors.append(
                        f"{scenario_id} 引用了未在 "
                        "feature_definitions_v1.json "
                        "中找到的 feature："
                        f"{feature_name}"
                    )

            if status != "ready":
                errors.append(
                    f"{scenario_id} "
                    "当前应标记为 ready"
                )

            if (
                scenario.get(
                    "standard_set_allowed"
                )
                is not True
            ):
                errors.append(
                    f"{scenario_id} "
                    "当前应允许进入 Standard Set"
                )

            if (
                scenario.get(
                    "hard_test_only"
                )
                is not False
            ):
                errors.append(
                    f"{scenario_id} "
                    "不应标记为 Hard Test only"
                )

            if (
                task_type
                == "experiment_diagnosis"
            ):
                issue_counter[
                    primary_issue
                ] += 1

                if (
                    primary_issue
                    not in EXPECTED_DIAGNOSIS_ISSUES
                ):
                    errors.append(
                        f"{scenario_id} 的 "
                        "primary_issue 非法："
                        f"{primary_issue}"
                    )

            else:
                if (
                    primary_issue
                    != "not_applicable"
                ):
                    errors.append(
                        f"{scenario_id} 的 "
                        "primary_issue 应为 "
                        "not_applicable"
                    )

    missing_diagnosis_issues = (
        EXPECTED_DIAGNOSIS_ISSUES
        - set(issue_counter)
    )

    if missing_diagnosis_issues:
        errors.append(
            "experiment_diagnosis "
            "缺少 primary issue 覆盖："
            f"{sorted(missing_diagnosis_issues)}"
        )

    deferred = scenario_config.get(
        "deferred_scenario_families",
        [],
    )

    if len(deferred) != 2:
        errors.append(
            "当前应保留 2 个 "
            "deferred scenario families"
        )

    for scenario in deferred:
        scenario_id = scenario.get(
            "scenario_family_id"
        )

        if not scenario_id:
            errors.append(
                "发现缺少 scenario_family_id "
                "的 deferred scenario"
            )
            continue

        all_ids.append(
            scenario_id
        )

        if (
            scenario.get("status")
            != "deferred"
        ):
            errors.append(
                f"{scenario_id} "
                "必须标记为 deferred"
            )

        if (
            scenario.get(
                "standard_set_allowed"
            )
            is not False
        ):
            errors.append(
                f"{scenario_id} "
                "不得进入 Standard Set"
            )

        if not scenario.get(
            "reason"
        ):
            errors.append(
                f"{scenario_id} "
                "缺少 deferred reason"
            )

    duplicate_ids = sorted(
        scenario_id
        for scenario_id, count
        in Counter(all_ids).items()
        if count > 1
    )

    if duplicate_ids:
        errors.append(
            "发现重复 scenario_family_id："
            f"{duplicate_ids}"
        )

    if errors:
        print(
            "DAY3 SCENARIO FAMILY VALIDATION FAILED"
        )

        for error in errors:
            print("-", error)

        raise SystemExit(1)

    print(
        "DAY3 SCENARIO FAMILY VALIDATION PASSED"
    )

    print(
        "配置文件：",
        SCENARIO_PATH,
    )

    print(
        "experiment_diagnosis scenario families：",
        active_counts[
            "experiment_diagnosis"
        ],
    )

    print(
        "metric_interpretation scenario families：",
        active_counts[
            "metric_interpretation"
        ],
    )

    print(
        "model_comparison scenario families：",
        active_counts[
            "model_comparison"
        ],
    )

    print(
        "active scenario families 总数：",
        sum(
            active_counts.values()
        ),
    )

    print(
        "deferred scenario families：",
        len(deferred),
    )

    print(
        "diagnosis primary issue 分布：",
        dict(
            sorted(
                issue_counter.items()
            )
        ),
    )


if __name__ == "__main__":
    main()
