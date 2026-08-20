import json
from collections import Counter
from pathlib import Path

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
    / "pilot_dataset_v1.jsonl"
)

TEMPLATE_PATH = (
    ROOT
    / "configs"
    / "template_families_v1.json"
)

EXPECTED_TOTAL = 144

EXPECTED_SPLITS = {
    "train": 72,
    "validation": 24,
    "test": 48,
}

EXPECTED_TASK_TYPES = {
    "experiment_diagnosis": 96,
    "metric_interpretation": 24,
    "model_comparison": 24,
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
                    f"Invalid JSONL at line "
                    f"{line_number}: {exc}"
                )

    return records


def extract_diagnosis_ground_truth(result):
    if (
        isinstance(result, dict)
        and isinstance(
            result.get("ground_truth"),
            dict,
        )
    ):
        return result[
            "ground_truth"
        ]

    return result


def recompute_ground_truth(sample):
    raw_record = sample[
        "raw_record"
    ]

    task_type = sample[
        "task_type"
    ]

    scenario_id = sample[
        "scenario_family_id"
    ]

    if task_type == "experiment_diagnosis":
        base_ground_truth = (
            extract_diagnosis_ground_truth(
                build_ground_truth(
                    raw_record
                )
            )
        )
    else:
        base_ground_truth = (
            build_non_diagnosis_ground_truth(
                raw_record
            )
        )

    return enrich_ground_truth_explanation(
        task_type=task_type,
        scenario_id=scenario_id,
        raw_record=raw_record,
        ground_truth=base_ground_truth,
    )


def canonical(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def get_nested_value(
    record,
    path,
):
    if "." not in path:
        return record[path]

    current = record

    for part in path.split("."):
        current = current[part]

    return current


def main():
    errors = []

    if not DATA_PATH.exists():
        print(
            "DAY3 PILOT DATASET VALIDATION FAILED"
        )
        print(
            "- Pilot Dataset（试生成数据集）不存在：",
            DATA_PATH,
        )
        raise SystemExit(1)

    try:
        samples = load_jsonl(
            DATA_PATH
        )

    except ValueError as exc:
        print(
            "DAY3 PILOT DATASET VALIDATION FAILED"
        )
        print("-", exc)
        raise SystemExit(1)

    template_config = load_json(
        TEMPLATE_PATH
    )

    template_map = {
        item[
            "template_family_id"
        ]: item
        for item in template_config[
            "template_families"
        ]
    }

    if len(samples) != EXPECTED_TOTAL:
        errors.append(
            "Pilot Sample（试生成样本）"
            f"应为 {EXPECTED_TOTAL}，"
            f"当前为 {len(samples)}"
        )

    sample_ids = [
        sample["sample_id"]
        for sample in samples
    ]

    if (
        len(sample_ids)
        != len(set(sample_ids))
    ):
        errors.append(
            "发现重复 Sample ID（样本 ID）"
        )

    template_ids = [
        sample[
            "template_family_id"
        ]
        for sample in samples
    ]

    if (
        len(template_ids)
        != len(set(template_ids))
    ):
        errors.append(
            "Pilot Set（试生成集）中 "
            "同一 Template Family（模板族）"
            "出现超过 1 条样本"
        )

    if (
        set(template_ids)
        != set(template_map)
    ):
        errors.append(
            "Pilot Set（试生成集）"
            "没有覆盖全部 Template Family（模板族）"
        )

    split_counter = Counter()
    task_counter = Counter()

    prompts = []

    for sample in samples:
        sample_id = sample[
            "sample_id"
        ]

        template_id = sample[
            "template_family_id"
        ]

        if (
            template_id
            not in template_map
        ):
            errors.append(
                f"{sample_id}: 未知 "
                f"Template Family（模板族）"
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
            sample["scenario_family_id"]
            != template[
                "scenario_family_id"
            ]
        ):
            errors.append(
                f"{sample_id}: Scenario Family（场景族）"
                "与 Template Family（模板族）不一致"
            )

        if (
            sample["task_type"]
            != template[
                "task_type"
            ]
        ):
            errors.append(
                f"{sample_id}: Task Type（任务类型）"
                "与 Template Family（模板族）不一致"
            )

        raw_record = sample[
            "raw_record"
        ]

        if "ground_truth" in raw_record:
            errors.append(
                f"{sample_id}: Raw Record（原始记录）"
                "不得包含 ground_truth"
            )

        if "primary_issue" in raw_record:
            errors.append(
                f"{sample_id}: Raw Record（原始记录）"
                "不得直接包含 primary_issue"
            )

        for required_input in template[
            "required_inputs"
        ]:
            try:
                get_nested_value(
                    raw_record,
                    required_input,
                )

            except (
                KeyError,
                TypeError,
            ):
                errors.append(
                    f"{sample_id}: 缺少 Required Input"
                    f"（必要输入） {required_input}"
                )

        gt_errors = validate_output(
            sample[
                "ground_truth"
            ]
        )

        if gt_errors:
            errors.append(
                f"{sample_id}: Output Schema"
                f"（输出结构）验证失败："
                + "; ".join(
                    gt_errors
                )
            )

        try:
            recomputed = (
                recompute_ground_truth(
                    sample
                )
            )

        except Exception as exc:
            errors.append(
                f"{sample_id}: Ground Truth Recompute"
                f"（标准答案重计算）失败：{exc}"
            )
            continue

        if (
            canonical(recomputed)
            != canonical(
                sample[
                    "ground_truth"
                ]
            )
        ):
            errors.append(
                f"{sample_id}: Ground Truth"
                "（标准答案）重计算结果不一致"
            )

        messages = sample.get(
            "messages",
            []
        )

        if len(messages) != 3:
            errors.append(
                f"{sample_id}: Messages（对话格式）"
                "必须包含 system/user/assistant 三条"
            )

        else:
            roles = [
                message.get("role")
                for message in messages
            ]

            if roles != [
                "system",
                "user",
                "assistant",
            ]:
                errors.append(
                    f"{sample_id}: Messages Role"
                    "（对话角色）顺序错误"
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
                    f"{sample_id}: User Message（用户消息）"
                    "与 Prompt（提示词）不一致"
                )

            try:
                assistant_json = json.loads(
                    messages[2][
                        "content"
                    ]
                )

                if (
                    canonical(
                        assistant_json
                    )
                    != canonical(
                        sample[
                            "ground_truth"
                        ]
                    )
                ):
                    errors.append(
                        f"{sample_id}: Assistant Target"
                        "（助手目标答案）"
                        "与 Ground Truth（标准答案）不一致"
                    )

            except json.JSONDecodeError:
                errors.append(
                    f"{sample_id}: Assistant Target"
                    "（助手目标答案）不是合法 JSON"
                )

        prompt = sample[
            "prompt"
        ]

        prompts.append(
            prompt.strip()
        )

        if (
            sample[
                "scenario_family_id"
            ]
            in prompt
        ):
            errors.append(
                f"{sample_id}: Prompt（提示词）"
                "泄漏 Scenario Family ID（场景族 ID）"
            )

        if (
            sample[
                "template_family_id"
            ]
            in prompt
        ):
            errors.append(
                f"{sample_id}: Prompt（提示词）"
                "泄漏 Template Family ID（模板族 ID）"
            )

        if (
            sample["task_type"]
            == "experiment_diagnosis"
            and sample[
                "primary_issue"
            ]
            in prompt
        ):
            errors.append(
                f"{sample_id}: Prompt（提示词）"
                "直接泄漏 Primary Issue（主要问题）"
            )

        split_counter[
            sample["split"]
        ] += 1

        task_counter[
            sample["task_type"]
        ] += 1

    if (
        len(prompts)
        != len(set(prompts))
    ):
        errors.append(
            "发现完全重复 Prompt（提示词）"
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
        != EXPECTED_TASK_TYPES
    ):
        errors.append(
            "Task Type Distribution（任务类型分布）"
            f"错误：{dict(task_counter)}"
        )

    split_files = {
        "train":
            ROOT
            / "data"
            / "generated"
            / "pilot_train_v1.jsonl",
        "validation":
            ROOT
            / "data"
            / "generated"
            / "pilot_validation_v1.jsonl",
        "test":
            ROOT
            / "data"
            / "generated"
            / "pilot_test_v1.jsonl",
    }

    for split_name, path in (
        split_files.items()
    ):
        if not path.exists():
            errors.append(
                f"缺少 Split JSONL（划分数据文件）："
                f"{path}"
            )
            continue

        split_records = load_jsonl(
            path
        )

        expected_ids = {
            sample["sample_id"]
            for sample in samples
            if sample["split"]
            == split_name
        }

        actual_ids = {
            sample["sample_id"]
            for sample in split_records
        }

        if actual_ids != expected_ids:
            errors.append(
                f"{split_name} Split JSONL（划分数据文件）"
                "与 Master JSONL（主数据文件）不一致"
            )

    if errors:
        print(
            "DAY3 PILOT DATASET VALIDATION FAILED"
        )

        for error in errors:
            print("-", error)

        raise SystemExit(1)

    print(
        "DAY3 PILOT DATASET VALIDATION PASSED"
    )

    print(
        "Pilot Samples（试生成样本）：",
        len(samples),
    )

    print(
        "Unique Template Families（唯一模板族）：",
        len(set(template_ids)),
    )

    print(
        "Unique Prompts（唯一提示词）：",
        len(set(prompts)),
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
        "Ground Truth Recompute Check"
        "（标准答案重计算检查）：PASSED"
    )

    print(
        "Template Leakage Check"
        "（模板泄漏检查）：PASSED"
    )


if __name__ == "__main__":
    main()
