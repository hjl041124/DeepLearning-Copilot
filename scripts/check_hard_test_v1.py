import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.hard_test_property_engine import (
    build_deterministic_ground_truth,
    compute_diagnosis_features,
    decision_signature,
)

from src.evaluation.output_validator import (
    validate_output,
)


ROOT = Path.cwd()

DATA_PATH = (
    ROOT
    / "data"
    / "generated"
    / "hard_test_v1.jsonl"
)

SPEC_PATH = (
    ROOT
    / "configs"
    / "hard_test_spec_v1.json"
)

STANDARD_PATH = (
    ROOT
    / "data"
    / "generated"
    / "full_dataset_v1.jsonl"
)


EXPECTED_TOTAL = 240

EXPECTED_PROPERTY_COUNTS = {
    "directional_boundary": 60,
    "invariance_distractor": 60,
    "priority_composition": 120,
}

EXPECTED_GROUP_COUNTS = {
    "HTG_DIRECTIONAL_BOUNDARY": 60,
    "HTG_INVARIANCE_DISTRACTOR": 60,
    "HTG_PRIORITY_COMPOSITION": 120,
}

EXPECTED_TASK_COUNTS = {
    "experiment_diagnosis": 220,
    "model_comparison": 20,
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


def canonical(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_family_map(spec):
    result = {}

    for group in spec[
        "hard_test_groups"
    ]:
        for family in group[
            "families"
        ]:
            result[
                family[
                    "hard_test_family_id"
                ]
            ] = {
                "group":
                    group,
                "family":
                    family,
            }

    return result


def strip_metadata(record):
    result = copy.deepcopy(
        record
    )

    result.pop(
        "metadata",
        None,
    )

    return result


def validate_messages(
    sample,
    errors,
):
    sample_id = sample[
        "sample_id"
    ]

    messages = sample.get(
        "messages",
        []
    )

    if len(messages) != 3:
        errors.append(
            f"{sample_id}: Messages（对话格式）必须有 3 条"
        )
        return

    roles = [
        message.get(
            "role"
        )
        for message in messages
    ]

    if roles != [
        "system",
        "user",
        "assistant",
    ]:
        errors.append(
            f"{sample_id}: Message Role（对话角色）顺序错误"
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
            f"{sample_id}: Prompt（提示词）与 User Message（用户消息）不一致"
        )

    try:
        assistant_target = json.loads(
            messages[2][
                "content"
            ]
        )

    except json.JSONDecodeError:
        errors.append(
            f"{sample_id}: Assistant Target（助手目标答案）不是合法 JSON"
        )
        return

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
            f"{sample_id}: Assistant Target（助手目标答案）与 Ground Truth（标准答案）不一致"
        )


def validate_directional_pair(
    family,
    pair_samples,
    errors,
):
    family_id = family[
        "hard_test_family_id"
    ]

    if len(
        pair_samples
    ) != 2:
        errors.append(
            f"{family_id}: Directional Pair（方向性样本对）必须有 2 条"
        )
        return

    member_map = {
        sample[
            "pair_member"
        ]:
            sample
        for sample in pair_samples
    }

    if (
        set(
            member_map
        )
        != {
            "below",
            "above",
        }
    ):
        errors.append(
            f"{family_id}: Directional Pair（方向性样本对）必须包含 below/above"
        )
        return

    below = member_map[
        "below"
    ]

    above = member_map[
        "above"
    ]

    if (
        below[
            "shared_context"
        ]
        != above[
            "shared_context"
        ]
    ):
        errors.append(
            f"{below['pair_id']}: Boundary Pair（边界样本对）的 Shared Context（共享背景）必须一致"
        )

    if (
        below[
            "render_mode"
        ]
        != above[
            "render_mode"
        ]
    ):
        errors.append(
            f"{below['pair_id']}: Boundary Pair（边界样本对）的 Render Mode（呈现形式）必须一致"
        )

    below_issue = (
        below[
            "ground_truth"
        ][
            "primary_issue"
        ]
    )

    above_issue = (
        above[
            "ground_truth"
        ][
            "primary_issue"
        ]
    )

    if (
        below_issue
        != family[
            "expected_below_primary_issue"
        ]
    ):
        errors.append(
            f"{below['pair_id']}: below Primary Issue（主要问题）错误"
        )

    if (
        above_issue
        != family[
            "expected_above_primary_issue"
        ]
    ):
        errors.append(
            f"{above['pair_id']}: above Primary Issue（主要问题）错误"
        )

    below_features = (
        compute_diagnosis_features(
            below[
                "raw_record"
            ]
        )
    )

    above_features = (
        compute_diagnosis_features(
            above[
                "raw_record"
            ]
        )
    )

    feature = family[
        "threshold_feature"
    ]

    threshold = float(
        below[
            "property_metadata"
        ][
            "threshold"
        ]
    )

    below_value = float(
        below_features[
            feature
        ]
    )

    above_value = float(
        above_features[
            feature
        ]
    )

    worse_direction = family[
        "worse_direction"
    ]

    if worse_direction == "higher":
        if not (
            below_value
            < threshold
            < above_value
        ):
            errors.append(
                f"{below['pair_id']}: Boundary Feature（边界特征）没有正确跨越阈值"
            )

    elif worse_direction == "lower":
        if not (
            below_value
            <= threshold
            < above_value
        ):
            errors.append(
                f"{below['pair_id']}: Lower-is-worse Boundary（越低越差边界）没有正确跨越阈值"
            )

    else:
        errors.append(
            f"{family_id}: 未知 worse_direction={worse_direction}"
        )


def validate_invariance_pair(
    family,
    pair_samples,
    errors,
):
    if len(
        pair_samples
    ) != 2:
        errors.append(
            f"{family['hard_test_family_id']}: Invariance Pair（不变性样本对）必须有 2 条"
        )
        return

    member_map = {
        sample[
            "pair_member"
        ]:
            sample
        for sample in pair_samples
    }

    if (
        set(
            member_map
        )
        != {
            "base",
            "perturbed",
        }
    ):
        errors.append(
            f"{family['hard_test_family_id']}: Invariance Pair（不变性样本对）必须包含 base/perturbed"
        )
        return

    base = member_map[
        "base"
    ]

    perturbed = member_map[
        "perturbed"
    ]

    if (
        base[
            "shared_context"
        ]
        != perturbed[
            "shared_context"
        ]
    ):
        errors.append(
            f"{base['pair_id']}: Invariance Pair（不变性样本对）的 Shared Context（共享背景）必须一致"
        )

    if (
        base[
            "render_mode"
        ]
        != perturbed[
            "render_mode"
        ]
    ):
        errors.append(
            f"{base['pair_id']}: Invariance Pair（不变性样本对）的 Render Mode（呈现形式）必须一致"
        )

    if (
        canonical(
            strip_metadata(
                perturbed[
                    "raw_record"
                ]
            )
        )
        != canonical(
            base[
                "raw_record"
            ]
        )
    ):
        errors.append(
            f"{base['pair_id']}: Invariance Perturbation（不变性扰动）改变了非 Metadata（元数据）字段"
        )

    if (
        decision_signature(
            base[
                "ground_truth"
            ]
        )
        != decision_signature(
            perturbed[
                "ground_truth"
            ]
        )
    ):
        errors.append(
            f"{base['pair_id']}: Invariance Decision Signature（不变性决策签名）发生变化"
        )

    if (
        base[
            "prompt"
        ]
        == perturbed[
            "prompt"
        ]
    ):
        errors.append(
            f"{base['pair_id']}: Invariance Pair（不变性样本对）的 Prompt（提示词）应该发生扰动"
        )


def main():
    errors = []

    try:
        samples = load_jsonl(
            DATA_PATH
        )

        spec = load_json(
            SPEC_PATH
        )

        standard_samples = load_jsonl(
            STANDARD_PATH
        )

    except Exception as exc:
        print(
            "DAY4 HARD TEST DATASET VALIDATION FAILED"
        )
        print(
            "-",
            exc,
        )
        raise SystemExit(1)

    family_map = build_family_map(
        spec
    )

    if len(
        samples
    ) != EXPECTED_TOTAL:
        errors.append(
            f"Hard Test（困难测试集）应为 {EXPECTED_TOTAL} 条，当前为 {len(samples)}"
        )

    sample_ids = [
        sample[
            "sample_id"
        ]
        for sample in samples
    ]

    if (
        len(
            sample_ids
        )
        != len(
            set(
                sample_ids
            )
        )
    ):
        errors.append(
            "Hard Test（困难测试集）存在重复 Sample ID（样本 ID）"
        )

    prompts = [
        sample[
            "prompt"
        ].strip()
        for sample in samples
    ]

    if (
        len(
            prompts
        )
        != len(
            set(
                prompts
            )
        )
    ):
        errors.append(
            "Hard Test（困难测试集）存在完全重复 Prompt（提示词）"
        )

    standard_prompts = {
        sample[
            "prompt"
        ].strip()
        for sample
        in standard_samples
    }

    overlap = (
        set(
            prompts
        )
        & standard_prompts
    )

    if overlap:
        errors.append(
            f"Hard Test（困难测试集）与 Standard Set（标准集）出现 {len(overlap)} 条完全重复 Prompt（提示词）"
        )

    family_counter = Counter()
    property_counter = Counter()
    group_counter = Counter()
    task_counter = Counter()

    pair_groups = defaultdict(
        list
    )

    for sample in samples:
        sample_id = sample[
            "sample_id"
        ]

        family_id = sample[
            "hard_test_family_id"
        ]

        if (
            family_id
            not in family_map
        ):
            errors.append(
                f"{sample_id}: 未知 Hard Test Family（困难测试族） {family_id}"
            )
            continue

        family_info = family_map[
            family_id
        ]

        family = family_info[
            "family"
        ]

        group = family_info[
            "group"
        ]

        if (
            sample[
                "hard_test_group_id"
            ]
            != group[
                "group_id"
            ]
        ):
            errors.append(
                f"{sample_id}: Hard Test Group（困难测试组）不一致"
            )

        if (
            sample[
                "test_property_type"
            ]
            != group[
                "test_property_type"
            ]
        ):
            errors.append(
                f"{sample_id}: Test Property Type（测试属性类型）不一致"
            )

        if (
            sample.get(
                "evaluation_only"
            )
            is not True
        ):
            errors.append(
                f"{sample_id}: Hard Test 必须为 Evaluation Only（仅评估）"
            )

        if (
            sample.get(
                "split"
            )
            != "hard_test"
        ):
            errors.append(
                f"{sample_id}: split 必须为 hard_test"
            )

        output_errors = validate_output(
            sample[
                "ground_truth"
            ]
        )

        if output_errors:
            errors.append(
                f"{sample_id}: Output Schema（输出结构）失败："
                + "; ".join(
                    output_errors
                )
            )

        try:
            recomputed = (
                build_deterministic_ground_truth(
                    sample[
                        "raw_record"
                    ]
                )
            )

        except Exception as exc:
            errors.append(
                f"{sample_id}: Ground Truth Recompute（标准答案重计算）失败：{exc}"
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
                f"{sample_id}: Ground Truth Recompute（标准答案重计算）不一致"
            )

        validate_messages(
            sample,
            errors,
        )

        if (
            family_id
            in sample[
                "prompt"
            ]
        ):
            errors.append(
                f"{sample_id}: Prompt（提示词）泄漏 Hard Test Family ID（困难测试族 ID）"
            )

        family_counter[
            family_id
        ] += 1

        property_counter[
            sample[
                "test_property_type"
            ]
        ] += 1

        group_counter[
            sample[
                "hard_test_group_id"
            ]
        ] += 1

        task_counter[
            sample[
                "task_type"
            ]
        ] += 1

        if (
            sample.get(
                "pair_id"
            )
            is not None
        ):
            pair_groups[
                sample[
                    "pair_id"
                ]
            ].append(
                sample
            )

        if (
            sample[
                "test_property_type"
            ]
            == "priority_composition"
        ):
            expected_issue = family[
                "expected_primary_issue"
            ]

            actual_issue = sample[
                "ground_truth"
            ][
                "primary_issue"
            ]

            if (
                actual_issue
                != expected_issue
            ):
                errors.append(
                    f"{sample_id}: Priority Composition（优先级组合）应为 {expected_issue}，当前为 {actual_issue}"
                )

    for (
        family_id,
        family_info,
    ) in family_map.items():
        expected_count = family_info[
            "family"
        ][
            "target_samples"
        ]

        actual_count = family_counter[
            family_id
        ]

        if (
            actual_count
            != expected_count
        ):
            errors.append(
                f"{family_id}: Hard Test Family Quota（困难测试族配额）应为 {expected_count}，当前为 {actual_count}"
            )

    if (
        dict(
            property_counter
        )
        != EXPECTED_PROPERTY_COUNTS
    ):
        errors.append(
            "Property Distribution（测试属性分布）错误："
            f"{dict(property_counter)}"
        )

    if (
        dict(
            group_counter
        )
        != EXPECTED_GROUP_COUNTS
    ):
        errors.append(
            "Group Distribution（困难测试组分布）错误："
            f"{dict(group_counter)}"
        )

    if (
        dict(
            task_counter
        )
        != EXPECTED_TASK_COUNTS
    ):
        errors.append(
            "Task Type Distribution（任务类型分布）错误："
            f"{dict(task_counter)}"
        )

    for (
        pair_id,
        pair_samples,
    ) in pair_groups.items():
        family_id = pair_samples[
            0
        ][
            "hard_test_family_id"
        ]

        family_info = family_map[
            family_id
        ]

        property_type = family_info[
            "group"
        ][
            "test_property_type"
        ]

        if (
            property_type
            == "directional_boundary"
        ):
            validate_directional_pair(
                family_info[
                    "family"
                ],
                pair_samples,
                errors,
            )

        elif (
            property_type
            == "invariance_distractor"
        ):
            validate_invariance_pair(
                family_info[
                    "family"
                ],
                pair_samples,
                errors,
            )

    expected_pair_count = 60

    if (
        len(
            pair_groups
        )
        != expected_pair_count
    ):
        errors.append(
            f"Pair-based Hard Test（成对困难测试）应有 {expected_pair_count} 个 Pair（样本对），当前为 {len(pair_groups)}"
        )

    if errors:
        print(
            "DAY4 HARD TEST DATASET VALIDATION FAILED"
        )

        print(
            "错误数量：",
            len(errors),
        )

        for error in errors[
            :60
        ]:
            print(
                "-",
                error,
            )

        if len(
            errors
        ) > 60:
            print(
                "... additional errors:",
                len(errors) - 60,
            )

        raise SystemExit(1)

    print(
        "DAY4 HARD TEST DATASET VALIDATION PASSED"
    )

    print(
        "Hard Test Samples（困难测试样本）：",
        len(samples),
    )

    print(
        "Unique Sample IDs（唯一样本 ID）：",
        len(
            set(
                sample_ids
            )
        ),
    )

    print(
        "Unique Prompts（唯一提示词）：",
        len(
            set(
                prompts
            )
        ),
    )

    print(
        "Hard Test Families（困难测试族）：",
        len(
            family_counter
        ),
    )

    print(
        "Unique Pairs（唯一测试样本对）：",
        len(
            pair_groups
        ),
    )

    print(
        "Property Distribution（测试属性分布）：",
        dict(
            property_counter
        ),
    )

    print(
        "Task Type Distribution（任务类型分布）：",
        dict(
            task_counter
        ),
    )

    print(
        "Ground Truth Recompute Check"
        "（标准答案重计算检查）：PASSED"
    )

    print(
        "Directional Pair Check"
        "（方向性样本对检查）：PASSED"
    )

    print(
        "Invariance Pair Check"
        "（不变性样本对检查）：PASSED"
    )

    print(
        "Priority Composition Check"
        "（优先级组合检查）：PASSED"
    )

    print(
        "Standard Set Overlap Check"
        "（标准数据集重叠检查）：PASSED"
    )


if __name__ == "__main__":
    main()
