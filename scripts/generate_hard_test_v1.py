import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scripts.generate_pilot_dataset_v1 import (
    SYSTEM_PROMPT,
    generate_diagnosis_record,
    generate_non_diagnosis_record,
    stable_seed,
)

from src.evaluation.dataset_text_renderer import (
    OUTPUT_INSTRUCTION,
)

from src.evaluation.hard_test_property_engine import (
    add_irrelevant_metadata,
    build_class_imbalance_boundary_pair,
    build_deterministic_ground_truth,
    build_generalization_boundary_pair,
    build_label_noise_boundary_pair,
    decision_signature,
    merge_diagnosis_records,
)

from src.evaluation.output_validator import (
    validate_output,
)


ROOT = Path.cwd()

SPEC_PATH = (
    ROOT
    / "configs"
    / "hard_test_spec_v1.json"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "generated"
    / "hard_test_v1.jsonl"
)

STATS_PATH = (
    ROOT
    / "data"
    / "generated"
    / "hard_test_v1.stats.json"
)


HARD_RENDER_MODES = [
    "challenge_note",
    "incident_report",
    "compact_log",
    "review_request",
]


BOUNDARY_BUILDERS = {
    "HT_DIR_GENERALIZATION_GAP":
        build_generalization_boundary_pair,

    "HT_DIR_CLASS_IMBALANCE_RATIO":
        build_class_imbalance_boundary_pair,

    "HT_DIR_LABEL_NOISE":
        build_label_noise_boundary_pair,
}


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def canonical(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_ground_truth(
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


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"

    if isinstance(value, list):
        return (
            "["
            + ", ".join(
                format_value(item)
                for item in value
            )
            + "]"
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(value)


def flatten_relevant_items(
    raw_record,
):
    items = []

    preferred_keys = [
        "metric_name",
        "metric_direction",
        "train_metric",
        "validation_metric",
        "reference_performance",
        "training_curve",
        "validation_curve",
        "class_counts",
        "per_class_metric",
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
        "primary_metric",
        "model_a_value",
        "model_b_value",
        "quality_metric",
        "quality_direction",
        "model_a_quality",
        "model_b_quality",
        "model_a_latency_ms",
        "model_b_latency_ms",
        "model_a_accuracy",
        "model_a_macro_f1",
        "model_b_accuracy",
        "model_b_macro_f1",
    ]

    for key in preferred_keys:
        if key in raw_record:
            items.append(
                (
                    key,
                    raw_record[key],
                )
            )

    data_quality = raw_record.get(
        "data_quality",
        {},
    )

    for key, value in data_quality.items():
        if float(value) != 0.0:
            items.append(
                (
                    f"data_quality.{key}",
                    value,
                )
            )

    flags = raw_record.get(
        "flags",
        {},
    )

    for key, value in flags.items():
        if value:
            items.append(
                (
                    f"flags.{key}",
                    value,
                )
            )

    return items


def question_for_task(
    task_type,
):
    if task_type == "experiment_diagnosis":
        return (
            "请判断这个实验最主要的问题，"
            "说明依据，并给出下一步建议。"
        )

    if task_type == "model_comparison":
        return (
            "请比较两个模型，并根据给定指标"
            "说明应该如何选择。"
        )

    if task_type == "metric_interpretation":
        return (
            "请解释这些指标中最值得关注的现象。"
        )

    raise ValueError(
        f"Unsupported task_type: {task_type}"
    )


def render_shared_context(
    shared_context,
):
    return (
        f"case_ref: {shared_context['case_ref']}\n"
        f"dataset_alias: {shared_context['dataset_alias']}"
    )


def render_metadata(
    metadata,
):
    if not metadata:
        return ""

    lines = [
        "Additional Metadata（附加元数据）:"
    ]

    for key, value in sorted(
        metadata.items()
    ):
        lines.append(
            f"- {key}: {format_value(value)}"
        )

    return "\n".join(lines)


def render_hard_prompt(
    task_type,
    raw_record,
    shared_context,
    render_mode,
    perturbation_metadata=None,
):
    items = flatten_relevant_items(
        raw_record
    )

    question = question_for_task(
        task_type
    )

    context_text = render_shared_context(
        shared_context
    )

    metadata_text = render_metadata(
        perturbation_metadata
    )

    if render_mode == "challenge_note":
        body = [
            "Experiment Challenge Note（实验挑战记录）",
            "",
            context_text,
            "",
        ]

        for key, value in items:
            body.append(
                f"{key}: {format_value(value)}"
            )

        if metadata_text:
            body.extend(
                [
                    "",
                    metadata_text,
                ]
            )

        body.extend(
            [
                "",
                question,
            ]
        )

        result = "\n".join(body)

    elif render_mode == "incident_report":
        observation_lines = [
            f"- {key}: {format_value(value)}"
            for key, value in items
        ]

        result = (
            "Experiment Incident Report（实验问题报告）\n\n"
            "Context（背景）:\n"
            f"{context_text}\n\n"
            "Observed Evidence（观测证据）:\n"
            + "\n".join(
                observation_lines
            )
        )

        if metadata_text:
            result += (
                "\n\n"
                + metadata_text
            )

        result += (
            "\n\n"
            "Request（分析请求）:\n"
            + question
        )

    elif render_mode == "compact_log":
        lines = [
            "[experiment_review]",
            f"case_ref={shared_context['case_ref']}",
            f"dataset_alias={shared_context['dataset_alias']}",
        ]

        for key, value in items:
            lines.append(
                f"{key}={format_value(value)}"
            )

        if perturbation_metadata:
            for key, value in sorted(
                perturbation_metadata.items()
            ):
                lines.append(
                    f"meta.{key}={format_value(value)}"
                )

        lines.extend(
            [
                "",
                question,
            ]
        )

        result = "\n".join(lines)

    elif render_mode == "review_request":
        fragments = [
            (
                f"{key} 为 "
                f"{format_value(value)}"
            )
            for key, value in items
        ]

        result = (
            "请帮我复核这次实验。"
            f"记录编号为 {shared_context['case_ref']}，"
            f"数据集代号为 {shared_context['dataset_alias']}。"
            + "；".join(fragments)
            + "。"
        )

        if metadata_text:
            result += (
                "\n\n"
                + metadata_text
            )

        result += (
            "\n\n"
            + question
        )

    else:
        raise ValueError(
            f"Unsupported render mode: {render_mode}"
        )

    return (
        result
        + "\n\n"
        + OUTPUT_INSTRUCTION
    )


def build_shared_context(
    family_id,
    index,
):
    seed = stable_seed(
        f"{family_id}::{index}::shared_context"
    )

    rng = random.Random(
        seed
    )

    pair_key = (
        f"{family_id}-{index}"
    )

    digest = hashlib.sha1(
        pair_key.encode(
            "utf-8"
        )
    ).hexdigest()[:8]

    return {
        "case_ref":
            f"HT-{digest}",

        "dataset_alias":
            rng.choice(
                [
                    "vision_set_alpha",
                    "classification_beta",
                    "experiment_gamma",
                    "benchmark_delta",
                    "dataset_epsilon",
                ]
            ),
    }


def build_distractor_metadata(
    field_names,
    family_id,
    pair_index,
):
    rng = random.Random(
        stable_seed(
            f"{family_id}::{pair_index}::metadata"
        )
    )

    result = {}

    for field_name in field_names:
        if field_name == "gpu_name":
            result[field_name] = rng.choice(
                [
                    "RTX 3090",
                    "RTX 4090",
                    "A100",
                    "L40S",
                    "H100",
                ]
            )

        elif field_name == "seed":
            result[field_name] = rng.randint(
                0,
                99999,
            )

        elif field_name == "run_name":
            result[field_name] = (
                f"run_{rng.randint(1000, 9999)}"
            )

        elif field_name == "batch_size":
            result[field_name] = rng.choice(
                [
                    16,
                    32,
                    64,
                    128,
                ]
            )

        elif field_name == "optimizer_name":
            result[field_name] = rng.choice(
                [
                    "AdamW",
                    "Adam",
                    "SGD",
                ]
            )

        elif field_name == "training_epochs":
            result[field_name] = rng.randint(
                10,
                200,
            )

        elif field_name == "framework_version":
            result[field_name] = rng.choice(
                [
                    "2.4",
                    "2.5",
                    "2.6",
                ]
            )

        elif field_name == "checkpoint_name":
            result[field_name] = (
                f"checkpoint_{rng.randint(100, 999)}"
            )

        else:
            result[field_name] = (
                f"value_{rng.randint(100, 999)}"
            )

    return result


def make_sample_id(
    family_id,
    logical_index,
    member=None,
):
    key = (
        f"{family_id}::{logical_index}::{member}"
    )

    return (
        "hard_"
        + hashlib.sha1(
            key.encode(
                "utf-8"
            )
        ).hexdigest()[:14]
    )


def make_messages(
    prompt,
    ground_truth,
):
    return [
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
            "content": json.dumps(
                ground_truth,
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def build_boundary_samples(
    group,
    family,
    epsilon_candidates,
):
    family_id = family[
        "hard_test_family_id"
    ]

    builder = BOUNDARY_BUILDERS[
        family_id
    ]

    samples = []

    for pair_index in range(
        family["pair_count"]
    ):
        epsilon = epsilon_candidates[
            pair_index
            % len(
                epsilon_candidates
            )
        ]

        pair = builder(
            epsilon
        )

        pair_id = (
            f"{family_id}_PAIR_{pair_index:02d}"
        )

        shared_context = build_shared_context(
            family_id,
            pair_index,
        )

        render_mode = HARD_RENDER_MODES[
            pair_index
            % len(
                HARD_RENDER_MODES
            )
        ]

        expected_map = {
            "below":
                family[
                    "expected_below_primary_issue"
                ],

            "above":
                family[
                    "expected_above_primary_issue"
                ],
        }

        for member in [
            "below",
            "above",
        ]:
            raw_record = pair[
                member
            ]

            ground_truth = (
                build_deterministic_ground_truth(
                    raw_record
                )
            )

            validate_ground_truth(
                ground_truth
            )

            expected_issue = expected_map[
                member
            ]

            if (
                ground_truth[
                    "primary_issue"
                ]
                != expected_issue
            ):
                raise ValueError(
                    f"{family_id} {pair_id} {member}: "
                    f"expected {expected_issue}, got "
                    f"{ground_truth['primary_issue']}"
                )

            prompt = render_hard_prompt(
                task_type=family[
                    "task_type"
                ],
                raw_record=raw_record,
                shared_context=shared_context,
                render_mode=render_mode,
            )

            sample_id = make_sample_id(
                family_id,
                pair_index,
                member,
            )

            samples.append(
                {
                    "sample_id":
                        sample_id,

                    "dataset_stage":
                        "hard_test_v1",

                    "split":
                        "hard_test",

                    "evaluation_only":
                        True,

                    "task_type":
                        family[
                            "task_type"
                        ],

                    "hard_test_group_id":
                        group[
                            "group_id"
                        ],

                    "hard_test_family_id":
                        family_id,

                    "test_property_type":
                        group[
                            "test_property_type"
                        ],

                    "pair_id":
                        pair_id,

                    "pair_member":
                        member,

                    "render_mode":
                        render_mode,

                    "shared_context":
                        shared_context,

                    "raw_record":
                        raw_record,

                    "prompt":
                        prompt,

                    "ground_truth":
                        ground_truth,

                    "messages":
                        make_messages(
                            prompt,
                            ground_truth,
                        ),

                    "property_metadata": {
                        "threshold_feature":
                            family[
                                "threshold_feature"
                            ],

                        "threshold":
                            pair[
                                "threshold"
                            ],

                        "epsilon":
                            epsilon,

                        "expected_primary_issue":
                            expected_issue,

                        "expected_relation":
                            "directional_threshold_crossing",
                    },
                }
            )

    return samples


def generate_base_record_for_invariance(
    family,
    pair_index,
):
    family_id = family[
        "hard_test_family_id"
    ]

    scenario_id = family[
        "base_scenario_family_ids"
    ][0]

    rng = random.Random(
        stable_seed(
            f"{family_id}::{pair_index}::base"
        )
    )

    task_type = family[
        "task_type"
    ]

    if task_type == "experiment_diagnosis":
        record = generate_diagnosis_record(
            scenario_id,
            rng,
        )

        record[
            "task_type"
        ] = "experiment_diagnosis"

        return record

    return generate_non_diagnosis_record(
        task_type,
        scenario_id,
        rng,
    )


def build_invariance_samples(
    group,
    family,
):
    family_id = family[
        "hard_test_family_id"
    ]

    samples = []

    for pair_index in range(
        family["pair_count"]
    ):
        base_record = (
            generate_base_record_for_invariance(
                family,
                pair_index,
            )
        )

        metadata = (
            build_distractor_metadata(
                family[
                    "perturbation_fields"
                ],
                family_id,
                pair_index,
            )
        )

        perturbed_record = (
            add_irrelevant_metadata(
                base_record,
                metadata,
            )
        )

        base_gt = (
            build_deterministic_ground_truth(
                base_record
            )
        )

        perturbed_gt = (
            build_deterministic_ground_truth(
                perturbed_record
            )
        )

        validate_ground_truth(
            base_gt
        )

        validate_ground_truth(
            perturbed_gt
        )

        if (
            decision_signature(
                base_gt
            )
            != decision_signature(
                perturbed_gt
            )
        ):
            raise ValueError(
                f"{family_id} pair {pair_index}: "
                "Invariance signature changed"
            )

        pair_id = (
            f"{family_id}_PAIR_{pair_index:02d}"
        )

        shared_context = build_shared_context(
            family_id,
            pair_index,
        )

        render_mode = HARD_RENDER_MODES[
            pair_index
            % len(
                HARD_RENDER_MODES
            )
        ]

        for member, raw_record, ground_truth in [
            (
                "base",
                base_record,
                base_gt,
            ),
            (
                "perturbed",
                perturbed_record,
                perturbed_gt,
            ),
        ]:
            perturbation_metadata = (
                metadata
                if member
                == "perturbed"
                else None
            )

            prompt = render_hard_prompt(
                task_type=family[
                    "task_type"
                ],
                raw_record=raw_record,
                shared_context=shared_context,
                render_mode=render_mode,
                perturbation_metadata=
                    perturbation_metadata,
            )

            sample_id = make_sample_id(
                family_id,
                pair_index,
                member,
            )

            samples.append(
                {
                    "sample_id":
                        sample_id,

                    "dataset_stage":
                        "hard_test_v1",

                    "split":
                        "hard_test",

                    "evaluation_only":
                        True,

                    "task_type":
                        family[
                            "task_type"
                        ],

                    "hard_test_group_id":
                        group[
                            "group_id"
                        ],

                    "hard_test_family_id":
                        family_id,

                    "test_property_type":
                        group[
                            "test_property_type"
                        ],

                    "pair_id":
                        pair_id,

                    "pair_member":
                        member,

                    "render_mode":
                        render_mode,

                    "shared_context":
                        shared_context,

                    "raw_record":
                        raw_record,

                    "prompt":
                        prompt,

                    "ground_truth":
                        ground_truth,

                    "messages":
                        make_messages(
                            prompt,
                            ground_truth,
                        ),

                    "property_metadata": {
                        "expected_relation":
                            "decision_signature_invariant",

                        "perturbation_fields":
                            family[
                                "perturbation_fields"
                            ],

                        "applied_metadata":
                            perturbation_metadata,
                    },
                }
            )

    return samples


def build_priority_record(
    family,
    sample_index,
):
    family_id = family[
        "hard_test_family_id"
    ]

    rng = random.Random(
        stable_seed(
            f"{family_id}::{sample_index}::priority"
        )
    )

    component_records = []

    for scenario_id in family[
        "component_scenario_family_ids"
    ]:
        record = generate_diagnosis_record(
            scenario_id,
            rng,
        )

        component_records.append(
            record
        )

    return merge_diagnosis_records(
        component_records
    )


def build_priority_samples(
    group,
    family,
):
    family_id = family[
        "hard_test_family_id"
    ]

    samples = []

    for sample_index in range(
        family[
            "target_samples"
        ]
    ):
        raw_record = build_priority_record(
            family,
            sample_index,
        )

        ground_truth = (
            build_deterministic_ground_truth(
                raw_record
            )
        )

        validate_ground_truth(
            ground_truth
        )

        expected_issue = family[
            "expected_primary_issue"
        ]

        if (
            ground_truth[
                "primary_issue"
            ]
            != expected_issue
        ):
            raise ValueError(
                f"{family_id} sample {sample_index}: "
                f"expected {expected_issue}, got "
                f"{ground_truth['primary_issue']}"
            )

        shared_context = build_shared_context(
            family_id,
            sample_index,
        )

        render_mode = HARD_RENDER_MODES[
            sample_index
            % len(
                HARD_RENDER_MODES
            )
        ]

        prompt = render_hard_prompt(
            task_type=family[
                "task_type"
            ],
            raw_record=raw_record,
            shared_context=shared_context,
            render_mode=render_mode,
        )

        sample_id = make_sample_id(
            family_id,
            sample_index,
            "single",
        )

        samples.append(
            {
                "sample_id":
                    sample_id,

                "dataset_stage":
                    "hard_test_v1",

                "split":
                    "hard_test",

                "evaluation_only":
                    True,

                "task_type":
                    family[
                        "task_type"
                    ],

                "hard_test_group_id":
                    group[
                        "group_id"
                    ],

                "hard_test_family_id":
                    family_id,

                "test_property_type":
                    group[
                        "test_property_type"
                    ],

                "pair_id":
                    None,

                "pair_member":
                    None,

                "render_mode":
                    render_mode,

                "shared_context":
                    shared_context,

                "raw_record":
                    raw_record,

                "prompt":
                    prompt,

                "ground_truth":
                    ground_truth,

                "messages":
                    make_messages(
                        prompt,
                        ground_truth,
                    ),

                "property_metadata": {
                    "expected_relation":
                        "rule_priority",

                    "component_scenario_family_ids":
                        family[
                            "component_scenario_family_ids"
                        ],

                    "expected_primary_issue":
                        expected_issue,
                },
            }
        )

    return samples


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
    spec = load_json(
        SPEC_PATH
    )

    epsilon_candidates = spec[
        "test_property_types"
    ][
        "directional_boundary"
    ][
        "epsilon_candidates"
    ]

    samples = []

    for group in spec[
        "hard_test_groups"
    ]:
        property_type = group[
            "test_property_type"
        ]

        for family in group[
            "families"
        ]:
            if (
                property_type
                == "directional_boundary"
            ):
                generated = (
                    build_boundary_samples(
                        group,
                        family,
                        epsilon_candidates,
                    )
                )

            elif (
                property_type
                == "invariance_distractor"
            ):
                generated = (
                    build_invariance_samples(
                        group,
                        family,
                    )
                )

            elif (
                property_type
                == "priority_composition"
            ):
                generated = (
                    build_priority_samples(
                        group,
                        family,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported property type: "
                    f"{property_type}"
                )

            if (
                len(generated)
                != family[
                    "target_samples"
                ]
            ):
                raise ValueError(
                    f"{family['hard_test_family_id']}: "
                    f"expected {family['target_samples']} "
                    f"samples, got {len(generated)}"
                )

            samples.extend(
                generated
            )

    samples.sort(
        key=lambda item: (
            item[
                "hard_test_group_id"
            ],
            item[
                "hard_test_family_id"
            ],
            item[
                "pair_id"
            ]
            or "",
            item[
                "sample_id"
            ],
        )
    )

    write_jsonl(
        OUTPUT_PATH,
        samples,
    )

    group_counter = Counter(
        sample[
            "hard_test_group_id"
        ]
        for sample in samples
    )

    property_counter = Counter(
        sample[
            "test_property_type"
        ]
        for sample in samples
    )

    family_counter = Counter(
        sample[
            "hard_test_family_id"
        ]
        for sample in samples
    )

    task_counter = Counter(
        sample[
            "task_type"
        ]
        for sample in samples
    )

    issue_counter = Counter(
        sample[
            "ground_truth"
        ][
            "primary_issue"
        ]
        for sample in samples
    )

    pair_ids = {
        sample[
            "pair_id"
        ]
        for sample in samples
        if sample[
            "pair_id"
        ]
        is not None
    }

    stats = {
        "total_samples":
            len(samples),

        "evaluation_only":
            True,

        "group_distribution":
            dict(
                group_counter
            ),

        "property_distribution":
            dict(
                property_counter
            ),

        "family_distribution":
            dict(
                family_counter
            ),

        "task_type_distribution":
            dict(
                task_counter
            ),

        "primary_issue_distribution":
            dict(
                issue_counter
            ),

        "unique_pair_count":
            len(pair_ids),

        "unique_prompt_count":
            len(
                {
                    sample[
                        "prompt"
                    ]
                    for sample in samples
                }
            ),
    }

    STATS_PATH.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "HARD TEST GENERATION PASSED"
    )

    print(
        "Hard Test Samples（困难测试样本）：",
        len(samples),
    )

    print(
        "Hard Test Families（困难测试族）：",
        len(family_counter),
    )

    print(
        "Unique Pairs（唯一测试样本对）：",
        len(pair_ids),
    )

    print(
        "Property Distribution（测试属性分布）：",
        dict(property_counter),
    )

    print(
        "Group Distribution（困难测试组分布）：",
        dict(group_counter),
    )

    print(
        "Task Type Distribution（任务类型分布）：",
        dict(task_counter),
    )


if __name__ == "__main__":
    main()
