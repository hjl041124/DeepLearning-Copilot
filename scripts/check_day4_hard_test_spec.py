import json
from collections import Counter
from pathlib import Path


ROOT = Path.cwd()

SPEC_PATH = ROOT / "configs" / "hard_test_spec_v1.json"
SCENARIO_PATH = ROOT / "configs" / "scenario_families_v1.json"
THRESHOLD_PATH = ROOT / "configs" / "threshold_bands_v1.json"
RULE_PATH = ROOT / "configs" / "diagnosis_rules_v1.json"
SURVEY_PATH = ROOT / "docs" / "DAY4_HARD_TEST_REFERENCE_SURVEY.md"


EXPECTED_REFERENCE_NAMES = {
    "CheckList",
    "Hypothesis",
    "Deepchecks",
    "HELM",
    "Robustness Gym",
}

EXPECTED_LICENSES = {
    "MIT",
    "MPL-2.0",
    "AGPL-3.0",
    "Apache-2.0",
}

EXPECTED_PROPERTY_TYPES = {
    "directional_boundary",
    "invariance_distractor",
    "priority_composition",
}

EXPECTED_GROUPS = {
    "HTG_DIRECTIONAL_BOUNDARY",
    "HTG_INVARIANCE_DISTRACTOR",
    "HTG_PRIORITY_COMPOSITION",
}

EXPECTED_PRIORITY = [
    "data_quality_issue",
    "optimization_problem",
    "class_imbalance",
    "overfitting",
    "underfitting",
    "no_clear_issue",
]

EXPECTED_TOTAL = 240
EXPECTED_FAMILIES = 12


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"缺少文件：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def active_scenario_ids(config):
    result = set()

    for task_config in config.get(
        "task_types",
        {},
    ).values():
        for scenario in task_config.get(
            "scenario_families",
            [],
        ):
            if scenario.get("status") == "ready":
                result.add(
                    scenario["scenario_family_id"]
                )

    return result


def main():
    errors = []

    try:
        spec = load_json(SPEC_PATH)
        scenarios = load_json(SCENARIO_PATH)
        thresholds = load_json(THRESHOLD_PATH)
        rules = load_json(RULE_PATH)

    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        print(
            "DAY4 HARD TEST SPEC VALIDATION FAILED"
        )
        print("-", exc)
        raise SystemExit(1)

    if not SURVEY_PATH.exists():
        errors.append(
            "缺少 Day 4 Reference Survey（参考调研）"
        )
    else:
        survey_text = SURVEY_PATH.read_text(
            encoding="utf-8"
        )

        required_survey_phrases = [
            "CheckList",
            "Hypothesis",
            "Deepchecks",
            "HELM",
            "Robustness Gym",
            "Directional Boundary Test",
            "Invariance Distractor Test",
            "Priority Composition Test",
            "LLM must not assign labels",
            "No third-party implementation code",
        ]

        for phrase in required_survey_phrases:
            if phrase not in survey_text:
                errors.append(
                    f"Reference Survey 缺少：{phrase}"
                )

    if spec.get("version") != "1.0":
        errors.append(
            "hard_test_spec_v1.json version 必须为 1.0"
        )

    policy = spec.get(
        "policy",
        {},
    )

    if policy.get("training_allowed") is not False:
        errors.append(
            "Hard Test 不得用于 Training（训练）"
        )

    if policy.get("validation_allowed") is not False:
        errors.append(
            "Hard Test 不得用于 Validation（验证）"
        )

    if policy.get("evaluation_only") is not True:
        errors.append(
            "Hard Test 必须为 Evaluation Only（仅评估）"
        )

    if (
        policy.get(
            "ground_truth_must_be_deterministic_python"
        )
        is not True
    ):
        errors.append(
            "Ground Truth（标准答案）必须由 Deterministic Python（确定性 Python）生成"
        )

    if (
        policy.get(
            "llm_must_not_assign_labels"
        )
        is not True
    ):
        errors.append(
            "LLM 不得决定 Ground Truth（标准答案）"
        )

    if (
        policy.get(
            "target_total_samples"
        )
        != EXPECTED_TOTAL
    ):
        errors.append(
            f"Hard Test 目标总数应为 {EXPECTED_TOTAL}"
        )

    references = spec.get(
        "references",
        [],
    )

    reference_names = {
        item.get("name")
        for item in references
    }

    if reference_names != EXPECTED_REFERENCE_NAMES:
        errors.append(
            "Reference（参考项目）集合不正确："
            f"{reference_names}"
        )

    licenses = {
        item.get("license")
        for item in references
    }

    if not EXPECTED_LICENSES.issubset(
        licenses
    ):
        errors.append(
            "License（许可证）记录不完整："
            f"{licenses}"
        )

    for reference in references:
        if reference.get(
            "copied_code"
        ) is not False:
            errors.append(
                f"{reference.get('name')} 必须明确 copied_code=false"
            )

    if (
        spec.get(
            "rule_priority"
        )
        != EXPECTED_PRIORITY
    ):
        errors.append(
            "Hard Test Rule Priority（规则优先级）与项目设计不一致"
        )

    rule_text = json.dumps(
        rules,
        ensure_ascii=False,
        sort_keys=True,
    )

    for issue in EXPECTED_PRIORITY:
        if issue not in rule_text:
            errors.append(
                f"diagnosis_rules_v1.json 中未发现 Rule Priority Issue（规则优先级问题）：{issue}"
            )

    property_types = set(
        spec.get(
            "test_property_types",
            {},
        ).keys()
    )

    if (
        property_types
        != EXPECTED_PROPERTY_TYPES
    ):
        errors.append(
            "Test Property Type（测试属性类型）不正确："
            f"{property_types}"
        )

    groups = spec.get(
        "hard_test_groups",
        [],
    )

    group_ids = {
        group.get("group_id")
        for group in groups
    }

    if group_ids != EXPECTED_GROUPS:
        errors.append(
            "Hard Test Group（困难测试组）不正确："
            f"{group_ids}"
        )

    scenario_ids = active_scenario_ids(
        scenarios
    )

    threshold_text = json.dumps(
        thresholds,
        ensure_ascii=False,
        sort_keys=True,
    )

    all_family_ids = []
    total_samples = 0
    group_sample_counter = {}
    property_counter = Counter()

    valid_primary_issues = set(
        EXPECTED_PRIORITY
    ) | {
        "not_applicable"
    }

    for group in groups:
        group_id = group[
            "group_id"
        ]

        property_type = group[
            "test_property_type"
        ]

        if (
            property_type
            not in EXPECTED_PROPERTY_TYPES
        ):
            errors.append(
                f"{group_id} 使用未知 Test Property（测试属性）：{property_type}"
            )

        families = group.get(
            "families",
            [],
        )

        group_total = 0

        for family in families:
            family_id = family.get(
                "hard_test_family_id"
            )

            if not family_id:
                errors.append(
                    f"{group_id} 中发现缺少 hard_test_family_id 的配置"
                )
                continue

            all_family_ids.append(
                family_id
            )

            sample_count = family.get(
                "target_samples"
            )

            if sample_count != 20:
                errors.append(
                    f"{family_id} target_samples 应为 20"
                )

            total_samples += (
                sample_count
                or 0
            )

            group_total += (
                sample_count
                or 0
            )

            property_counter[
                property_type
            ] += (
                sample_count
                or 0
            )

            if property_type in {
                "directional_boundary",
                "invariance_distractor",
            }:
                pair_count = family.get(
                    "pair_count"
                )

                if pair_count != 10:
                    errors.append(
                        f"{family_id} pair_count 应为 10"
                    )

                if (
                    sample_count is not None
                    and sample_count % 2 != 0
                ):
                    errors.append(
                        f"{family_id} Pair-based Test（成对测试）样本数必须为偶数"
                    )

            if property_type == "directional_boundary":
                threshold_feature = family.get(
                    "threshold_feature"
                )

                if not threshold_feature:
                    errors.append(
                        f"{family_id} 缺少 threshold_feature"
                    )

                elif (
                    threshold_feature
                    not in threshold_text
                ):
                    errors.append(
                        f"{family_id} 引用的 Threshold Feature（阈值特征）未在 threshold_bands_v1.json 中找到："
                        f"{threshold_feature}"
                    )

                for field_name in [
                    "expected_below_primary_issue",
                    "expected_above_primary_issue",
                ]:
                    value = family.get(
                        field_name
                    )

                    if (
                        value
                        not in valid_primary_issues
                    ):
                        errors.append(
                            f"{family_id} 的 {field_name} 非法：{value}"
                        )

            base_ids = family.get(
                "base_scenario_family_ids",
                [],
            )

            component_ids = family.get(
                "component_scenario_family_ids",
                [],
            )

            referenced_ids = (
                base_ids
                + component_ids
            )

            for scenario_id in referenced_ids:
                if (
                    scenario_id
                    not in scenario_ids
                ):
                    errors.append(
                        f"{family_id} 引用了不存在或非 Ready 的 Scenario Family（场景族）："
                        f"{scenario_id}"
                    )

            if (
                property_type
                == "priority_composition"
            ):
                if len(
                    component_ids
                ) < 2:
                    errors.append(
                        f"{family_id} Priority Composition（优先级组合测试）至少需要两个 component scenario"
                    )

                expected_issue = family.get(
                    "expected_primary_issue"
                )

                if (
                    expected_issue
                    not in valid_primary_issues
                ):
                    errors.append(
                        f"{family_id} expected_primary_issue 非法：{expected_issue}"
                    )

        group_sample_counter[
            group_id
        ] = group_total

        if (
            group_total
            != group.get(
                "target_samples"
            )
        ):
            errors.append(
                f"{group_id} target_samples 与 Family 配额不一致"
            )

    if (
        len(all_family_ids)
        != EXPECTED_FAMILIES
    ):
        errors.append(
            f"Hard Test Family（困难测试族）应为 {EXPECTED_FAMILIES} 个，"
            f"当前为 {len(all_family_ids)}"
        )

    duplicate_ids = [
        family_id
        for family_id, count
        in Counter(
            all_family_ids
        ).items()
        if count > 1
    ]

    if duplicate_ids:
        errors.append(
            "发现重复 Hard Test Family ID（困难测试族 ID）："
            f"{duplicate_ids}"
        )

    if total_samples != EXPECTED_TOTAL:
        errors.append(
            f"Hard Test 总样本应为 {EXPECTED_TOTAL}，当前为 {total_samples}"
        )

    expected_property_distribution = {
        "directional_boundary": 60,
        "invariance_distractor": 60,
        "priority_composition": 120,
    }

    if (
        dict(
            property_counter
        )
        != expected_property_distribution
    ):
        errors.append(
            "Test Property（测试属性）目标样本分布不正确："
            f"{dict(property_counter)}"
        )

    if errors:
        print(
            "DAY4 HARD TEST SPEC VALIDATION FAILED"
        )

        for error in errors:
            print(
                "-",
                error,
            )

        raise SystemExit(1)

    print(
        "DAY4 HARD TEST SPEC VALIDATION PASSED"
    )

    print(
        "Reference Projects（参考项目）：",
        len(references),
    )

    print(
        "Test Property Types（测试属性类型）：",
        len(property_types),
    )

    print(
        "Hard Test Groups（困难测试组）：",
        len(groups),
    )

    print(
        "Hard Test Families（困难测试族）：",
        len(all_family_ids),
    )

    print(
        "Target Samples（目标样本）：",
        total_samples,
    )

    print(
        "Property Distribution（测试属性分布）：",
        dict(
            property_counter
        ),
    )

    print(
        "Group Distribution（困难测试组分布）：",
        group_sample_counter,
    )


if __name__ == "__main__":
    main()
