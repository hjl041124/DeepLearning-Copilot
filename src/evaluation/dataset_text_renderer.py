import hashlib
import json


OUTPUT_INSTRUCTION = """
请严格输出一个 JSON 对象，并且只能包含以下字段：

task_type
primary_issue
severity
evidence_codes
recommended_action_codes
explanation
""".strip()


FIELD_NAMES = {
    "train_metric": "训练指标（train_metric）",
    "validation_metric": "验证指标（validation_metric）",
    "metric_direction": "指标方向（metric_direction）",
    "reference_performance": "参考性能（reference_performance）",
    "training_curve": "训练曲线（training_curve）",
    "validation_curve": "验证曲线（validation_curve）",
    "class_counts": "各类别样本量（class_counts）",
    "per_class_metric": "逐类别指标（per_class_metric）",
    "accuracy": "准确率（accuracy）",
    "macro_f1": "宏平均 F1（macro_f1）",
    "precision": "精确率（precision）",
    "recall": "召回率（recall）",
    "primary_metric": "主要指标（primary_metric）",
    "model_a_value": "Model A 指标值",
    "model_b_value": "Model B 指标值",
    "quality_metric": "质量指标（quality_metric）",
    "quality_direction": "质量指标方向（quality_direction）",
    "model_a_quality": "Model A 质量指标",
    "model_b_quality": "Model B 质量指标",
    "model_a_latency_ms": "Model A 延迟（ms）",
    "model_b_latency_ms": "Model B 延迟（ms）",
    "model_a_accuracy": "Model A accuracy",
    "model_a_macro_f1": "Model A macro-F1",
    "model_b_accuracy": "Model B accuracy",
    "model_b_macro_f1": "Model B macro-F1",
    "data_quality.label_noise_rate":
        "标签噪声率（label_noise_rate）",
    "data_quality.duplicate_rate":
        "重复样本率（duplicate_rate）",
    "data_quality.split_overlap_rate":
        "跨划分重叠率（split_overlap_rate）",
    "flags": "实验状态标记（flags）",
}


def _get_nested_value(record, path):
    if "." not in path:
        return record[path]

    current = record

    for part in path.split("."):
        current = current[part]

    return current


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.3f}"

    if isinstance(value, list):
        return "[" + ", ".join(
            _format_value(item)
            for item in value
        ) + "]"

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(value)


def _field_name(name):
    return FIELD_NAMES.get(
        name,
        name,
    )


def _question(task_type):
    if task_type == "experiment_diagnosis":
        return (
            "请判断这个实验最主要的问题是什么，"
            "说明支持判断的证据，并给出下一步建议。"
        )

    if task_type == "metric_interpretation":
        return (
            "请解释这些指标最值得关注的现象，"
            "并说明这些数值意味着什么。"
        )

    if task_type == "model_comparison":
        return (
            "请比较两个模型，并根据给出的实验指标"
            "说明应该如何做出选择。"
        )

    raise ValueError(
        f"Unsupported task_type: {task_type}"
    )


def _observation_items(
    raw_record,
    required_inputs,
):
    return [
        (
            input_name,
            _get_nested_value(
                raw_record,
                input_name,
            ),
        )
        for input_name in required_inputs
    ]


def render_prompt_v2(
    task_type,
    render_mode,
    raw_record,
    required_inputs,
    sample_id,
):
    items = _observation_items(
        raw_record,
        required_inputs,
    )

    question = _question(
        task_type
    )

    if render_mode == "structured_block":
        lines = [
            "实验快照（Experiment Snapshot）",
        ]

        for key, value in items:
            lines.append(
                f"- {_field_name(key)}: "
                f"{_format_value(value)}"
            )

        body = "\n".join(lines)

    elif render_mode == "tabular_report":
        rows = [
            "| 实验字段 | 数值 / 状态 |",
            "| --- | --- |",
        ]

        for key, value in items:
            rows.append(
                f"| {_field_name(key)} | "
                f"{_format_value(value)} |"
            )

        body = (
            "实验指标报告（Experiment Metrics Report）\n\n"
            + "\n".join(rows)
        )

    elif render_mode == "tracker_export":
        lines = [
            "[run_summary]",
            f"run_id={sample_id}",
        ]

        for key, value in items:
            lines.append(
                f"{key}={_format_value(value)}"
            )

        body = (
            "下面是一段 Experiment Tracker"
            "（实验跟踪器）的运行摘要：\n\n"
            + "\n".join(lines)
        )

    elif render_mode == "concise_note":
        fragments = []

        for key, value in items:
            fragments.append(
                f"{_field_name(key)}为"
                f"{_format_value(value)}"
            )

        body = (
            "我在检查一次深度学习实验。"
            + "，".join(fragments)
            + "。"
        )

    elif render_mode == "debug_ticket":
        observations = "\n".join(
            f"- {_field_name(key)}: "
            f"{_format_value(value)}"
            for key, value in items
        )

        body = (
            "Debug Ticket（调试工单）\n\n"
            "Context（背景）:\n"
            "- 模型训练或评估结果需要进一步分析。\n\n"
            "Observations（观测结果）:\n"
            f"{observations}\n\n"
            "Question（问题）:\n"
            f"{question}"
        )

    elif render_mode == "narrative_summary":
        fragments = [
            (
                f"{_field_name(key)}记录为"
                f"{_format_value(value)}"
            )
            for key, value in items
        ]

        body = (
            "在最近一次实验运行中，"
            + "；".join(fragments)
            + "。这些结果需要进一步分析。"
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


def _numeric_values(value):
    if isinstance(value, dict):
        value = value.values()

    return [
        float(item)
        for item in value
    ]


def _relative_gap(a, b):
    denominator = max(
        abs(float(a)),
        abs(float(b)),
        1e-12,
    )

    return abs(
        float(a) - float(b)
    ) / denominator


def _variant_index(
    scenario_id,
    raw_record,
    count,
):
    payload = (
        scenario_id
        + json.dumps(
            raw_record,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    ) % count


def _choose(
    scenario_id,
    raw_record,
    variants,
):
    index = _variant_index(
        scenario_id,
        raw_record,
        len(variants),
    )

    return variants[index]


def enrich_ground_truth_explanation(
    task_type,
    scenario_id,
    raw_record,
    ground_truth,
):
    output = dict(
        ground_truth
    )

    explanation = None

    if scenario_id == "ED_OF_GENERALIZATION_GAP_LATE_DEGRADATION":
        train = float(
            raw_record["train_metric"]
        )
        val = float(
            raw_record["validation_metric"]
        )
        curve = raw_record[
            "validation_curve"
        ]
        best = max(curve)
        gap = _relative_gap(
            train,
            val,
        )

        explanation = _choose(
            scenario_id,
            raw_record,
            [
                (
                    f"Training performance reaches {train:.3f}, "
                    f"while validation performance ends at {val:.3f}. "
                    f"The relative gap is about {gap:.3f}, and validation "
                    f"performance falls from a best value of {best:.3f}. "
                    "This is strong evidence of overfitting."
                ),
                (
                    f"The training result ({train:.3f}) is substantially "
                    f"better than the final validation result ({val:.3f}). "
                    f"Validation previously reached {best:.3f} and then "
                    "degraded, so the issue is not simply low overall "
                    "capacity but poor generalization."
                ),
            ],
        )

    elif scenario_id == "ED_OF_VALIDATION_LOSS_RISES":
        train = float(
            raw_record["train_metric"]
        )
        final_val = float(
            raw_record["validation_metric"]
        )
        best_val = min(
            raw_record["validation_curve"]
        )

        explanation = (
            f"Training loss is {train:.3f}, while validation loss "
            f"reaches a minimum near {best_val:.3f} and later rises "
            f"to {final_val:.3f}. The late validation deterioration "
            "while training remains strong is characteristic of overfitting."
        )

    elif scenario_id == "ED_UF_LOW_SCORES_VS_REFERENCE":
        train = float(
            raw_record["train_metric"]
        )
        val = float(
            raw_record["validation_metric"]
        )
        ref = float(
            raw_record["reference_performance"]
        )

        explanation = (
            f"Training ({train:.3f}) and validation ({val:.3f}) "
            f"remain close to each other, but both are well below "
            f"the explicit reference performance of {ref:.3f}. "
            "The small generalization gap together with the large "
            "reference shortfall supports underfitting."
        )

    elif scenario_id == "ED_UF_REFERENCE_SHORTFALL_PLATEAU":
        train = float(
            raw_record["train_metric"]
        )
        val = float(
            raw_record["validation_metric"]
        )
        ref = float(
            raw_record["reference_performance"]
        )

        explanation = (
            f"Training and validation performance remain close "
            f"({train:.3f} vs {val:.3f}) but are substantially below "
            f"the reference value {ref:.3f}. The training curve also "
            "shows only very small late improvements, providing "
            "additional evidence of underfitting."
        )

    elif scenario_id == "ED_OP_STRONG_LOSS_OSCILLATION":
        curve = _numeric_values(
            raw_record["training_curve"]
        )

        explanation = (
            f"Training loss repeatedly moves between approximately "
            f"{min(curve):.3f} and {max(curve):.3f} instead of "
            "decreasing smoothly. The large repeated swings indicate "
            "unstable optimization rather than normal small fluctuations."
        )

    elif scenario_id == "ED_OP_NAN_OR_INF":
        explanation = (
            "The training run explicitly contains a NaN or Inf signal. "
            "This is a direct numerical-instability indicator, so the "
            "primary issue is an optimization problem rather than a "
            "normal convergence pattern."
        )

    elif scenario_id in {
        "ED_CI_SKEW_WITH_CLASSWISE_COLLAPSE",
        "ED_CI_MINORITY_PERFORMANCE_COLLAPSE",
    }:
        counts = _numeric_values(
            raw_record["class_counts"]
        )
        metrics = _numeric_values(
            raw_record["per_class_metric"]
        )

        ratio = min(counts) / max(counts)
        metric_gap = max(metrics) - min(metrics)

        explanation = _choose(
            scenario_id,
            raw_record,
            [
                (
                    f"The smallest class contains only {int(min(counts))} "
                    f"samples compared with {int(max(counts))} in the "
                    f"largest class, giving a min/max ratio of {ratio:.3f}. "
                    f"Class-level performance also differs by {metric_gap:.3f}. "
                    "The distribution skew is therefore accompanied by "
                    "clear performance damage."
                ),
                (
                    f"Class support is strongly uneven "
                    f"({int(min(counts))} vs {int(max(counts))} samples), "
                    f"and the class-wise metric gap reaches {metric_gap:.3f}. "
                    "This indicates that the minority-class imbalance is "
                    "affecting predictive performance."
                ),
            ],
        )

    elif scenario_id == "ED_CI_ACCURACY_MASKS_MACRO_F1":
        counts = _numeric_values(
            raw_record["class_counts"]
        )
        accuracy = float(
            raw_record["accuracy"]
        )
        macro_f1 = float(
            raw_record["macro_f1"]
        )
        ratio = min(counts) / max(counts)
        metric_gap = accuracy - macro_f1

        explanation = _choose(
            scenario_id,
            raw_record,
            [
                (
                    f"The smallest class has {int(min(counts))} samples "
                    f"while the largest has {int(max(counts))}, giving "
                    f"a class-count ratio of only {ratio:.3f}. "
                    f"Accuracy is high at {accuracy:.3f}, but macro-F1 "
                    f"is only {macro_f1:.3f}, a gap of {metric_gap:.3f}. "
                    "The aggregate accuracy is masking weak class-balanced "
                    "performance."
                ),
                (
                    f"Although accuracy reaches {accuracy:.3f}, macro-F1 "
                    f"drops to {macro_f1:.3f}. Combined with the strongly "
                    f"skewed class counts ({int(min(counts))} to "
                    f"{int(max(counts))}), this suggests the model performs "
                    "well on dominant classes while underrepresented classes "
                    "are not being handled equally well."
                ),
            ],
        )

    elif scenario_id == "ED_DQ_HIGH_LABEL_NOISE":
        rate = float(
            raw_record[
                "data_quality"
            ][
                "label_noise_rate"
            ]
        )

        explanation = (
            f"The measured label-noise rate is {rate:.3f}, which lies "
            "inside the project's clearly abnormal sampling region. "
            "The main concern is therefore data quality rather than "
            "model capacity or normal optimization behavior."
        )

    elif scenario_id == "ED_DQ_HIGH_DUPLICATE_RATE":
        rate = float(
            raw_record[
                "data_quality"
            ][
                "duplicate_rate"
            ]
        )

        explanation = (
            f"The duplicate-sample rate is {rate:.3f}, which is "
            "clearly elevated under the project rules. A large amount "
            "of duplicated data can distort effective sample diversity "
            "and should be cleaned before interpreting model behavior."
        )

    elif scenario_id == "ED_DQ_SPLIT_OVERLAP":
        rate = float(
            raw_record[
                "data_quality"
            ][
                "split_overlap_rate"
            ]
        )

        explanation = (
            f"The measured train-validation overlap rate is {rate:.3f}. "
            "Because evaluation samples appear across data splits, the "
            "reported validation result may be overly optimistic and "
            "the split should be repaired."
        )

    elif scenario_id == "ED_DQ_PREPROCESSING_MISMATCH":
        explanation = (
            "The run explicitly reports inconsistent preprocessing "
            "between data splits. This can create artificial train-"
            "validation differences, so preprocessing alignment should "
            "be fixed before drawing conclusions about the model."
        )

    elif scenario_id == "ED_DQ_EXPLICIT_DISTRIBUTION_SHIFT":
        explanation = (
            "A deterministic distribution-shift flag is present. "
            "The training and evaluation data therefore should not be "
            "treated as coming from the same distribution, making "
            "data quality and dataset compatibility the primary concern."
        )

    elif scenario_id == "ED_NC_STABLE_CONVERGENCE_SMALL_GAP":
        train = float(
            raw_record["train_metric"]
        )
        val = float(
            raw_record["validation_metric"]
        )
        gap = _relative_gap(
            train,
            val,
        )

        explanation = (
            f"Training and validation performance are close "
            f"({train:.3f} vs {val:.3f}), with a relative gap of "
            f"about {gap:.3f}. Both curves improve smoothly and there "
            "is no strong degradation, oscillation, or data-quality "
            "signal, so no clear issue is detected."
        )

    elif scenario_id == "ED_NC_BALANCED_CLASSWISE_PERFORMANCE":
        counts = _numeric_values(
            raw_record["class_counts"]
        )
        metrics = _numeric_values(
            raw_record["per_class_metric"]
        )

        explanation = (
            f"Class counts are similar ({int(min(counts))} to "
            f"{int(max(counts))}) and class-wise performance differs "
            f"by only {max(metrics) - min(metrics):.3f}. Accuracy and "
            "macro-F1 are also close, so there is no strong evidence "
            "of class imbalance or another clear diagnostic issue."
        )

    elif scenario_id == "MI_ACCURACY_MACRO_F1_GAP":
        accuracy = float(
            raw_record["accuracy"]
        )
        macro_f1 = float(
            raw_record["macro_f1"]
        )

        explanation = (
            f"Accuracy is {accuracy:.3f}, while macro-F1 is only "
            f"{macro_f1:.3f}. The {accuracy - macro_f1:.3f} gap "
            "indicates that aggregate accuracy may be hiding uneven "
            "performance across classes."
        )

    elif scenario_id == "MI_PRECISION_RECALL_TRADEOFF":
        precision = float(
            raw_record["precision"]
        )
        recall = float(
            raw_record["recall"]
        )

        if precision > recall:
            explanation = (
                f"Precision ({precision:.3f}) is substantially higher "
                f"than recall ({recall:.3f}). The model makes fewer "
                "false-positive predictions but is likely missing a "
                "larger fraction of true positives."
            )
        else:
            explanation = (
                f"Recall ({recall:.3f}) is substantially higher than "
                f"precision ({precision:.3f}). The model captures more "
                "true positives, but this comes with more false-positive "
                "predictions."
            )

    elif scenario_id == "MI_CLASSWISE_PERFORMANCE_GAP":
        values = _numeric_values(
            raw_record["per_class_metric"]
        )

        explanation = (
            f"The best and worst class-specific results differ by "
            f"{max(values) - min(values):.3f}. Aggregate metrics alone "
            "would therefore hide a meaningful class-level performance gap."
        )

    elif scenario_id == "MI_TRAIN_VALIDATION_GAP":
        train = float(
            raw_record["train_metric"]
        )
        val = float(
            raw_record["validation_metric"]
        )

        explanation = (
            f"Training performance is {train:.3f} while validation "
            f"performance is {val:.3f}. The large direction-aware gap "
            "shows that performance on the training data does not "
            "transfer equally well to validation data."
        )

    elif scenario_id == "MC_CLEAR_QUALITY_WINNER":
        a = float(
            raw_record["model_a_value"]
        )
        b = float(
            raw_record["model_b_value"]
        )

        winner = (
            "Model A"
            if a > b
            else "Model B"
        )

        explanation = (
            f"Model A scores {a:.3f} and Model B scores {b:.3f} "
            f"on the stated primary metric. {winner} has a clearly "
            "better result under the requested comparison objective."
        )

    elif scenario_id == "MC_QUALITY_EFFICIENCY_TRADEOFF":
        aq = float(
            raw_record["model_a_quality"]
        )
        bq = float(
            raw_record["model_b_quality"]
        )
        al = float(
            raw_record["model_a_latency_ms"]
        )
        bl = float(
            raw_record["model_b_latency_ms"]
        )

        explanation = (
            f"Model A has quality {aq:.3f} with {al:.1f} ms latency, "
            f"while Model B has quality {bq:.3f} with {bl:.1f} ms "
            "latency. One model provides stronger predictive quality "
            "while the other is faster, so the final choice depends "
            "on deployment constraints."
        )

    elif scenario_id == "MC_IMBALANCED_METRIC_COMPARISON":
        aa = float(
            raw_record["model_a_accuracy"]
        )
        af = float(
            raw_record["model_a_macro_f1"]
        )
        ba = float(
            raw_record["model_b_accuracy"]
        )
        bf = float(
            raw_record["model_b_macro_f1"]
        )

        winner = (
            "Model A"
            if af > bf
            else "Model B"
        )

        explanation = (
            f"Model A reports accuracy/macro-F1 of {aa:.3f}/{af:.3f}, "
            f"while Model B reports {ba:.3f}/{bf:.3f}. Under an "
            f"imbalance-aware objective, {winner} is preferable because "
            "its macro-F1 is higher even if raw accuracy alone suggests "
            "a different conclusion."
        )

    elif scenario_id == "MC_NO_CLEAR_WINNER":
        a = float(
            raw_record["model_a_value"]
        )
        b = float(
            raw_record["model_b_value"]
        )

        gap = _relative_gap(
            a,
            b,
        )

        explanation = (
            f"Model A scores {a:.3f} and Model B scores {b:.3f}; "
            f"their relative difference is only about {gap:.3f}. "
            "That difference is too small for a confident winner under "
            "the current project rule, so more evidence is appropriate."
        )

    if explanation is not None:
        output[
            "explanation"
        ] = explanation

    return output
