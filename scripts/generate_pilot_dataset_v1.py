import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from src.evaluation.ground_truth_builder import build_ground_truth
from src.evaluation.non_diagnosis_ground_truth import (
    build_non_diagnosis_ground_truth,
)
from src.evaluation.output_validator import validate_output
from src.evaluation.dataset_text_renderer import (
    enrich_ground_truth_explanation,
    render_prompt_v2,
)


ROOT = Path.cwd()

TEMPLATE_PATH = ROOT / "configs" / "template_families_v1.json"
SAMPLING_PATH = ROOT / "configs" / "sampling_plan_v1.json"

OUTPUT_PATH = ROOT / "data" / "generated" / "pilot_dataset_v1.jsonl"

SPLIT_PATHS = {
    "train": ROOT / "data" / "generated" / "pilot_train_v1.jsonl",
    "validation": ROOT / "data" / "generated" / "pilot_validation_v1.jsonl",
    "test": ROOT / "data" / "generated" / "pilot_test_v1.jsonl",
}


SYSTEM_PROMPT = (
    "你是 DeepLearning-Copilot，一个深度学习实验分析助手。"
    "只能依据用户提供的实验数据进行判断，不得编造不存在的指标。"
    "请严格输出 JSON，不要输出 JSON 之外的额外文字。"
)


OUTPUT_INSTRUCTION = """
请严格输出一个 JSON 对象，并且只能包含以下字段：

task_type
primary_issue
severity
evidence_codes
recommended_action_codes
explanation
""".strip()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:16],
        16,
    )


def r3(value):
    return round(float(value), 3)


def neutral_data_quality():
    return {
        "label_noise_rate": 0.0,
        "duplicate_rate": 0.0,
        "missing_value_rate": 0.0,
        "corrupted_sample_rate": 0.0,
        "split_overlap_rate": 0.0,
    }


def neutral_flags():
    return {
        "nan_or_inf": False,
        "gradient_instability": False,
        "loss_divergence": False,
        "preprocessing_mismatch": False,
        "distribution_shift_detected": False,
    }


def diagnosis_base():
    return {
        "task_type": "experiment_diagnosis",
        "data_quality": neutral_data_quality(),
        "flags": neutral_flags(),
    }


def generate_diagnosis_record(
    scenario_id,
    rng,
):
    record = diagnosis_base()

    # --------------------------------------------------------
    # Overfitting（过拟合）
    # --------------------------------------------------------

    if (
        scenario_id
        == "ED_OF_GENERALIZATION_GAP_LATE_DEGRADATION"
    ):
        train_final = rng.uniform(
            0.93,
            0.97,
        )

        validation_final = rng.uniform(
            0.66,
            0.71,
        )

        best_validation = rng.uniform(
            0.80,
            0.84,
        )

        record.update(
            {
                "metric_name": "accuracy",
                "metric_direction": "higher_is_better",
                "train_metric": r3(train_final),
                "validation_metric": r3(validation_final),
                "validation_curve": [
                    r3(best_validation - 0.18),
                    r3(best_validation - 0.11),
                    r3(best_validation - 0.05),
                    r3(best_validation),
                    r3(best_validation - 0.03),
                    r3(best_validation - 0.07),
                    r3(validation_final),
                ],
            }
        )

        return record

    if (
        scenario_id
        == "ED_OF_VALIDATION_LOSS_RISES"
    ):
        train_loss = rng.uniform(
            0.14,
            0.19,
        )

        final_val_loss = rng.uniform(
            0.40,
            0.46,
        )

        best_val_loss = rng.uniform(
            0.26,
            0.30,
        )

        record.update(
            {
                "metric_name": "loss",
                "metric_direction": "lower_is_better",
                "train_metric": r3(train_loss),
                "validation_metric": r3(final_val_loss),
                "validation_curve": [
                    r3(best_val_loss + 0.34),
                    r3(best_val_loss + 0.22),
                    r3(best_val_loss + 0.11),
                    r3(best_val_loss),
                    r3(best_val_loss + 0.04),
                    r3(best_val_loss + 0.09),
                    r3(final_val_loss),
                ],
            }
        )

        return record

    # --------------------------------------------------------
    # Underfitting（欠拟合）
    # --------------------------------------------------------

    if (
        scenario_id
        == "ED_UF_LOW_SCORES_VS_REFERENCE"
    ):
        validation = rng.uniform(
            0.54,
            0.60,
        )

        train = validation + rng.uniform(
            0.01,
            0.025,
        )

        reference = rng.uniform(
            0.82,
            0.88,
        )

        record.update(
            {
                "metric_name": "accuracy",
                "metric_direction": "higher_is_better",
                "train_metric": r3(train),
                "validation_metric": r3(validation),
                "reference_performance": r3(reference),
            }
        )

        return record

    if (
        scenario_id
        == "ED_UF_REFERENCE_SHORTFALL_PLATEAU"
    ):
        validation = rng.uniform(
            0.55,
            0.59,
        )

        train = validation + rng.uniform(
            0.01,
            0.025,
        )

        reference = rng.uniform(
            0.84,
            0.89,
        )

        start = train - 0.035

        record.update(
            {
                "metric_name": "accuracy",
                "metric_direction": "higher_is_better",
                "train_metric": r3(train),
                "validation_metric": r3(validation),
                "reference_performance": r3(reference),
                "training_curve": [
                    r3(start),
                    r3(train - 0.018),
                    r3(train - 0.010),
                    r3(train - 0.006),
                    r3(train - 0.004),
                    r3(train - 0.003),
                    r3(train - 0.002),
                    r3(train - 0.001),
                    r3(train),
                ],
            }
        )

        return record

    # --------------------------------------------------------
    # Optimization Problem（优化问题）
    # --------------------------------------------------------

    if (
        scenario_id
        == "ED_OP_STRONG_LOSS_OSCILLATION"
    ):
        high = rng.uniform(
            1.05,
            1.25,
        )

        low = rng.uniform(
            0.45,
            0.60,
        )

        record.update(
            {
                "metric_name": "loss",
                "metric_direction": "lower_is_better",
                "training_curve": [
                    r3(high),
                    r3(low),
                    r3(high - 0.04),
                    r3(low + 0.02),
                    r3(high - 0.08),
                    r3(low + 0.04),
                    r3(high - 0.12),
                    r3(low + 0.05),
                ],
            }
        )

        return record

    if (
        scenario_id
        == "ED_OP_NAN_OR_INF"
    ):
        record["flags"]["nan_or_inf"] = True

        return record

    # --------------------------------------------------------
    # Class Imbalance（类别不平衡）
    # --------------------------------------------------------

    if (
        scenario_id
        == "ED_CI_SKEW_WITH_CLASSWISE_COLLAPSE"
    ):
        record.update(
            {
                "class_counts": [
                    rng.randint(950, 1200),
                    rng.randint(35, 75),
                    rng.randint(650, 850),
                ],
                "per_class_metric": [
                    r3(rng.uniform(0.88, 0.94)),
                    r3(rng.uniform(0.38, 0.52)),
                    r3(rng.uniform(0.80, 0.88)),
                ],
            }
        )

        return record

    if (
        scenario_id
        == "ED_CI_ACCURACY_MASKS_MACRO_F1"
    ):
        record.update(
            {
                "class_counts": [
                    rng.randint(1000, 1300),
                    rng.randint(40, 80),
                    rng.randint(600, 800),
                ],
                "accuracy": r3(
                    rng.uniform(0.89, 0.94)
                ),
                "macro_f1": r3(
                    rng.uniform(0.62, 0.70)
                ),
            }
        )

        return record

    if (
        scenario_id
        == "ED_CI_MINORITY_PERFORMANCE_COLLAPSE"
    ):
        record.update(
            {
                "class_counts": [
                    rng.randint(1100, 1400),
                    rng.randint(25, 55),
                    rng.randint(550, 750),
                ],
                "per_class_metric": [
                    r3(rng.uniform(0.89, 0.94)),
                    r3(rng.uniform(0.30, 0.45)),
                    r3(rng.uniform(0.78, 0.87)),
                ],
            }
        )

        return record

    # --------------------------------------------------------
    # Data Quality Issue（数据质量问题）
    # --------------------------------------------------------

    if (
        scenario_id
        == "ED_DQ_HIGH_LABEL_NOISE"
    ):
        record["data_quality"][
            "label_noise_rate"
        ] = r3(
            rng.uniform(0.24, 0.32)
        )

        return record

    if (
        scenario_id
        == "ED_DQ_HIGH_DUPLICATE_RATE"
    ):
        record["data_quality"][
            "duplicate_rate"
        ] = r3(
            rng.uniform(0.24, 0.32)
        )

        return record

    if (
        scenario_id
        == "ED_DQ_SPLIT_OVERLAP"
    ):
        record["data_quality"][
            "split_overlap_rate"
        ] = r3(
            rng.uniform(0.02, 0.04)
        )

        return record

    if (
        scenario_id
        == "ED_DQ_PREPROCESSING_MISMATCH"
    ):
        record["flags"][
            "preprocessing_mismatch"
        ] = True

        return record

    if (
        scenario_id
        == "ED_DQ_EXPLICIT_DISTRIBUTION_SHIFT"
    ):
        record["flags"][
            "distribution_shift_detected"
        ] = True

        return record

    # --------------------------------------------------------
    # No Clear Issue（无明显问题）
    # --------------------------------------------------------

    if (
        scenario_id
        == "ED_NC_STABLE_CONVERGENCE_SMALL_GAP"
    ):
        validation = rng.uniform(
            0.81,
            0.85,
        )

        train = validation + rng.uniform(
            0.01,
            0.025,
        )

        record.update(
            {
                "metric_name": "accuracy",
                "metric_direction": "higher_is_better",
                "train_metric": r3(train),
                "validation_metric": r3(validation),
                "training_curve": [
                    r3(train - 0.18),
                    r3(train - 0.12),
                    r3(train - 0.075),
                    r3(train - 0.040),
                    r3(train - 0.020),
                    r3(train - 0.009),
                    r3(train),
                ],
                "validation_curve": [
                    r3(validation - 0.17),
                    r3(validation - 0.11),
                    r3(validation - 0.070),
                    r3(validation - 0.036),
                    r3(validation - 0.017),
                    r3(validation - 0.007),
                    r3(validation),
                ],
            }
        )

        return record

    if (
        scenario_id
        == "ED_NC_BALANCED_CLASSWISE_PERFORMANCE"
    ):
        base_count = rng.randint(
            450,
            600,
        )

        record.update(
            {
                "class_counts": [
                    base_count,
                    base_count + rng.randint(-20, 20),
                    base_count + rng.randint(-20, 20),
                ],
                "per_class_metric": [
                    r3(rng.uniform(0.80, 0.84)),
                    r3(rng.uniform(0.80, 0.84)),
                    r3(rng.uniform(0.80, 0.84)),
                ],
                "accuracy": r3(
                    rng.uniform(0.81, 0.84)
                ),
                "macro_f1": r3(
                    rng.uniform(0.80, 0.83)
                ),
            }
        )

        return record

    raise ValueError(
        f"Unsupported diagnosis scenario: {scenario_id}"
    )


def generate_non_diagnosis_record(
    task_type,
    scenario_id,
    rng,
):
    if (
        scenario_id
        == "MI_ACCURACY_MACRO_F1_GAP"
    ):
        return {
            "task_type": task_type,
            "scenario_family_id": scenario_id,
            "accuracy": r3(
                rng.uniform(0.88, 0.93)
            ),
            "macro_f1": r3(
                rng.uniform(0.62, 0.70)
            ),
            "class_counts": {
                "class_A": rng.randint(900, 1200),
                "class_B": rng.randint(50, 100),
                "class_C": rng.randint(400, 700),
            },
        }

    if (
        scenario_id
        == "MI_PRECISION_RECALL_TRADEOFF"
    ):
        if rng.random() < 0.5:
            precision = rng.uniform(
                0.88,
                0.94,
            )

            recall = rng.uniform(
                0.55,
                0.67,
            )

        else:
            recall = rng.uniform(
                0.88,
                0.94,
            )

            precision = rng.uniform(
                0.55,
                0.67,
            )

        return {
            "task_type": task_type,
            "scenario_family_id": scenario_id,
            "precision": r3(precision),
            "recall": r3(recall),
        }

    if (
        scenario_id
        == "MI_CLASSWISE_PERFORMANCE_GAP"
    ):
        return {
            "task_type": task_type,
            "scenario_family_id": scenario_id,
            "per_class_metric": {
                "class_A": r3(
                    rng.uniform(0.88, 0.94)
                ),
                "class_B": r3(
                    rng.uniform(0.42, 0.55)
                ),
                "class_C": r3(
                    rng.uniform(0.78, 0.86)
                ),
            },
        }

    if (
        scenario_id
        == "MI_TRAIN_VALIDATION_GAP"
    ):
        return {
            "task_type": task_type,
            "scenario_family_id": scenario_id,
            "train_metric": r3(
                rng.uniform(0.90, 0.95)
            ),
            "validation_metric": r3(
                rng.uniform(0.66, 0.73)
            ),
            "metric_direction": "higher_is_better",
        }

    if (
        scenario_id
        == "MC_CLEAR_QUALITY_WINNER"
    ):
        model_a = rng.uniform(
            0.88,
            0.93,
        )

        model_b = rng.uniform(
            0.76,
            0.82,
        )

        if rng.random() < 0.5:
            model_a, model_b = (
                model_b,
                model_a,
            )

        return {
            "task_type": task_type,
            "scenario_family_id": scenario_id,
            "primary_metric": "macro_f1",
            "metric_direction": "higher_is_better",
            "model_a_value": r3(model_a),
            "model_b_value": r3(model_b),
        }

    if (
        scenario_id
        == "MC_QUALITY_EFFICIENCY_TRADEOFF"
    ):
        if rng.random() < 0.5:
            a_quality = rng.uniform(
                0.89,
                0.93,
            )
            b_quality = rng.uniform(
                0.77,
                0.82,
            )

            a_latency = rng.uniform(
                105,
                140,
            )
            b_latency = rng.uniform(
                45,
                70,
            )

        else:
            b_quality = rng.uniform(
                0.89,
                0.93,
            )
            a_quality = rng.uniform(
                0.77,
                0.82,
            )

            b_latency = rng.uniform(
                105,
                140,
            )
            a_latency = rng.uniform(
                45,
                70,
            )

        return {
            "task_type": task_type,
            "scenario_family_id": scenario_id,
            "quality_metric": "macro_f1",
            "quality_direction": "higher_is_better",
            "model_a_quality": r3(a_quality),
            "model_b_quality": r3(b_quality),
            "model_a_latency_ms": r3(a_latency),
            "model_b_latency_ms": r3(b_latency),
        }

    if (
        scenario_id
        == "MC_IMBALANCED_METRIC_COMPARISON"
    ):
        a_macro = rng.uniform(
            0.80,
            0.86,
        )

        b_macro = rng.uniform(
            0.68,
            0.75,
        )

        if rng.random() < 0.5:
            a_macro, b_macro = (
                b_macro,
                a_macro,
            )

        return {
            "task_type": task_type,
            "scenario_family_id": scenario_id,
            "model_a_accuracy": r3(
                rng.uniform(0.86, 0.92)
            ),
            "model_a_macro_f1": r3(
                a_macro
            ),
            "model_b_accuracy": r3(
                rng.uniform(0.86, 0.92)
            ),
            "model_b_macro_f1": r3(
                b_macro
            ),
        }

    if (
        scenario_id
        == "MC_NO_CLEAR_WINNER"
    ):
        center = rng.uniform(
            0.82,
            0.90,
        )

        delta = rng.uniform(
            0.002,
            0.010,
        )

        return {
            "task_type": task_type,
            "scenario_family_id": scenario_id,
            "primary_metric": "macro_f1",
            "metric_direction": "higher_is_better",
            "model_a_value": r3(
                center
            ),
            "model_b_value": r3(
                center - delta
            ),
        }

    raise ValueError(
        f"Unsupported non-diagnosis scenario: {scenario_id}"
    )


def extract_diagnosis_ground_truth(result):
    if (
        isinstance(result, dict)
        and isinstance(
            result.get("ground_truth"),
            dict,
        )
    ):
        return result["ground_truth"]

    required = {
        "task_type",
        "primary_issue",
        "severity",
        "evidence_codes",
        "recommended_action_codes",
        "explanation",
    }

    if (
        isinstance(result, dict)
        and required.issubset(result)
    ):
        return result

    raise ValueError(
        "Unexpected build_ground_truth() return structure"
    )


def build_ground_truth_for_sample(
    task_type,
    raw_record,
):
    if task_type == "experiment_diagnosis":
        result = build_ground_truth(
            raw_record
        )

        return extract_diagnosis_ground_truth(
            result
        )

    return build_non_diagnosis_ground_truth(
        raw_record
    )


def validate_ground_truth_or_raise(
    ground_truth,
):
    errors = validate_output(
        ground_truth
    )

    if errors:
        raise ValueError(
            "Ground Truth validation failed: "
            + "; ".join(errors)
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


def format_value(value):
    if isinstance(
        value,
        (dict, list),
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value)


def observation_items(
    raw_record,
    required_inputs,
):
    items = []

    for input_name in required_inputs:
        value = get_nested_value(
            raw_record,
            input_name,
        )

        items.append(
            (
                input_name,
                value,
            )
        )

    return items


def task_question(task_type):
    if (
        task_type
        == "experiment_diagnosis"
    ):
        return (
            "请根据这些实验结果诊断最主要的实验问题，"
            "给出证据和下一步建议。"
        )

    if (
        task_type
        == "metric_interpretation"
    ):
        return (
            "请解释这些指标反映了什么，并指出最值得关注的现象。"
        )

    if (
        task_type
        == "model_comparison"
    ):
        return (
            "请比较两个模型，并根据给定指标说明应如何做出选择。"
        )

    raise ValueError(
        f"Unsupported task_type: {task_type}"
    )


def render_prompt(
    task_type,
    render_mode,
    raw_record,
    required_inputs,
    sample_id,
):
    items = observation_items(
        raw_record,
        required_inputs,
    )

    question = task_question(
        task_type
    )

    if render_mode == "structured_block":
        lines = [
            "Experiment Snapshot",
            f"task_type: {task_type}",
        ]

        for key, value in items:
            lines.append(
                f"{key}: {format_value(value)}"
            )

        body = "\n".join(lines)

    elif render_mode == "tabular_report":
        rows = [
            "| Field | Value |",
            "| --- | --- |",
            f"| task_type | {task_type} |",
        ]

        for key, value in items:
            rows.append(
                f"| {key} | {format_value(value)} |"
            )

        body = (
            "Experiment Metrics Report\n\n"
            + "\n".join(rows)
        )

    elif render_mode == "tracker_export":
        lines = [
            "[run_summary]",
            f"run_id={sample_id}",
            f"task_type={task_type}",
        ]

        for key, value in items:
            lines.append(
                f"{key}={format_value(value)}"
            )

        body = "\n".join(lines)

    elif render_mode == "concise_note":
        fragments = [
            f"本次任务类型为 {task_type}。"
        ]

        for key, value in items:
            fragments.append(
                f"{key} 为 {format_value(value)}。"
            )

        body = " ".join(
            fragments
        )

    elif render_mode == "debug_ticket":
        observations = "\n".join(
            f"- {key}: {format_value(value)}"
            for key, value in items
        )

        body = (
            "Debug Ticket\n\n"
            "Context:\n"
            f"- task_type: {task_type}\n\n"
            "Observations:\n"
            f"{observations}\n\n"
            "Question:\n"
            f"{question}"
        )

    elif render_mode == "narrative_summary":
        fragments = []

        for key, value in items:
            fragments.append(
                f"{key} 记录为 {format_value(value)}"
            )

        body = (
            f"这是一次 {task_type} 任务。"
            "实验记录显示，"
            + "，".join(fragments)
            + "。"
        )

    else:
        raise ValueError(
            f"Unsupported render_mode: {render_mode}"
        )

    if render_mode != "debug_ticket":
        body += (
            "\n\n"
            + question
        )

    return (
        body
        + "\n\n"
        + OUTPUT_INSTRUCTION
    )


def build_sample(
    quota,
    template,
):
    template_id = quota[
        "template_family_id"
    ]

    scenario_id = quota[
        "scenario_family_id"
    ]

    task_type = quota[
        "task_type"
    ]

    expected_issue = quota[
        "primary_issue"
    ]

    rng = random.Random(
        stable_seed(template_id)
    )

    if (
        task_type
        == "experiment_diagnosis"
    ):
        raw_record = (
            generate_diagnosis_record(
                scenario_id,
                rng,
            )
        )

        raw_record[
            "scenario_family_id"
        ] = scenario_id

    else:
        raw_record = (
            generate_non_diagnosis_record(
                task_type,
                scenario_id,
                rng,
            )
        )

    ground_truth = (
        build_ground_truth_for_sample(
            task_type,
            raw_record,
        )
    )

    ground_truth = enrich_ground_truth_explanation(
        task_type=task_type,
        scenario_id=scenario_id,
        raw_record=raw_record,
        ground_truth=ground_truth,
    )

    validate_ground_truth_or_raise(
        ground_truth
    )

    if (
        task_type
        == "experiment_diagnosis"
        and ground_truth["primary_issue"]
        != expected_issue
    ):
        raise ValueError(
            f"{template_id}: expected "
            f"{expected_issue}, got "
            f"{ground_truth['primary_issue']}"
        )

    if (
        task_type
        != "experiment_diagnosis"
        and ground_truth["primary_issue"]
        != "not_applicable"
    ):
        raise ValueError(
            f"{template_id}: non-diagnosis "
            "primary_issue must be not_applicable"
        )

    sample_id = (
        "pilot_"
        + hashlib.sha1(
            template_id.encode(
                "utf-8"
            )
        ).hexdigest()[:12]
    )

    prompt = render_prompt_v2(
        task_type=task_type,
        render_mode=template[
            "render_mode"
        ],
        raw_record=raw_record,
        required_inputs=template[
            "required_inputs"
        ],
        sample_id=sample_id,
    )

    assistant_target = json.dumps(
        ground_truth,
        ensure_ascii=False,
        sort_keys=True,
    )

    return {
        "sample_id": sample_id,
        "dataset_stage": "pilot",
        "split": quota["split"],
        "task_type": task_type,
        "primary_issue":
            ground_truth["primary_issue"],
        "scenario_family_id":
            scenario_id,
        "template_family_id":
            template_id,
        "presentation_family_id":
            quota[
                "presentation_family_id"
            ],
        "render_mode":
            template["render_mode"],
        "raw_record":
            raw_record,
        "prompt":
            prompt,
        "ground_truth":
            ground_truth,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": assistant_target,
            },
        ],
        "metadata": {
            "generator_version": "pilot_v1",
            "deterministic_seed":
                stable_seed(
                    template_id
                ),
            "pilot_sample_count_for_template":
                quota[
                    "pilot_sample_count"
                ],
        },
    }


def write_jsonl(
    path,
    records,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    template_config = load_json(
        TEMPLATE_PATH
    )

    sampling_plan = load_json(
        SAMPLING_PATH
    )

    template_map = {
        template[
            "template_family_id"
        ]: template
        for template
        in template_config[
            "template_families"
        ]
    }

    pilot_quotas = [
        quota
        for quota
        in sampling_plan[
            "template_quotas"
        ]
        if quota[
            "pilot_sample_count"
        ] > 0
    ]

    samples = []

    for quota in pilot_quotas:
        template_id = quota[
            "template_family_id"
        ]

        template = template_map[
            template_id
        ]

        for _ in range(
            quota[
                "pilot_sample_count"
            ]
        ):
            samples.append(
                build_sample(
                    quota,
                    template,
                )
            )

    samples.sort(
        key=lambda x: (
            x["split"],
            x["task_type"],
            x["scenario_family_id"],
            x["template_family_id"],
        )
    )

    write_jsonl(
        OUTPUT_PATH,
        samples,
    )

    split_counter = Counter(
        sample["split"]
        for sample in samples
    )

    task_counter = Counter(
        sample["task_type"]
        for sample in samples
    )

    for split_name, path in (
        SPLIT_PATHS.items()
    ):
        split_samples = [
            sample
            for sample in samples
            if sample["split"]
            == split_name
        ]

        write_jsonl(
            path,
            split_samples,
        )

    print(
        "PILOT DATASET GENERATION PASSED"
    )

    print(
        "Master JSONL（主数据文件）：",
        OUTPUT_PATH,
    )

    print(
        "Pilot Samples（试生成样本）：",
        len(samples),
    )

    print(
        "Split Distribution（数据划分分布）：",
        dict(split_counter),
    )

    print(
        "Task Type Distribution（任务类型分布）：",
        dict(task_counter),
    )


if __name__ == "__main__":
    main()
