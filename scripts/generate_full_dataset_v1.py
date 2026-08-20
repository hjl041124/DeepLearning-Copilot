import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scripts.generate_pilot_dataset_v1 import (
    SYSTEM_PROMPT,
    build_ground_truth_for_sample,
    generate_diagnosis_record,
    generate_non_diagnosis_record,
    stable_seed,
    validate_ground_truth_or_raise,
)

from src.evaluation.dataset_text_renderer import (
    enrich_ground_truth_explanation,
    render_prompt_v2,
)


ROOT = Path.cwd()

TEMPLATE_PATH = (
    ROOT
    / "configs"
    / "template_families_v1.json"
)

SAMPLING_PATH = (
    ROOT
    / "configs"
    / "sampling_plan_v1.json"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "generated"
    / "full_dataset_v1.jsonl"
)

STATS_PATH = (
    ROOT
    / "data"
    / "generated"
    / "full_dataset_v1.stats.json"
)

SPLIT_PATHS = {
    "train":
        ROOT
        / "data"
        / "generated"
        / "full_train_v1.jsonl",

    "validation":
        ROOT
        / "data"
        / "generated"
        / "full_validation_v1.jsonl",

    "test":
        ROOT
        / "data"
        / "generated"
        / "full_test_v1.jsonl",
}


MODEL_NAMES = [
    "small_cnn",
    "resnet18",
    "convnext_tiny",
    "efficientnet_b0",
    "mlp_classifier",
]

DATASET_NAMES = [
    "dataset_alpha",
    "dataset_beta",
    "dataset_gamma",
    "dataset_delta",
    "dataset_epsilon",
]

OPTIMIZERS = [
    "AdamW",
    "Adam",
    "SGD",
]

BATCH_SIZES = [
    16,
    32,
    48,
    64,
    128,
]


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def build_run_context(
    sample_id,
    rng,
):
    learning_rate = 10 ** rng.uniform(
        -4.5,
        -3.0,
    )

    return {
        "run_ref": sample_id[-10:],
        "dataset": rng.choice(
            DATASET_NAMES
        ),
        "model": rng.choice(
            MODEL_NAMES
        ),
        "optimizer": rng.choice(
            OPTIMIZERS
        ),
        "batch_size": rng.choice(
            BATCH_SIZES
        ),
        "learning_rate": round(
            learning_rate,
            6,
        ),
    }


def format_run_context(
    context,
):
    return (
        "Run Context（运行背景）:\n"
        f"- run_ref: {context['run_ref']}\n"
        f"- dataset: {context['dataset']}\n"
        f"- model: {context['model']}\n"
        f"- optimizer: {context['optimizer']}\n"
        f"- batch_size: {context['batch_size']}\n"
        f"- learning_rate: {context['learning_rate']}"
    )


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


def build_full_sample(
    quota,
    template,
    variant_index,
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

    variant_key = (
        f"{template_id}::"
        f"{variant_index}"
    )

    numeric_seed = stable_seed(
        variant_key
    )

    context_seed = stable_seed(
        variant_key
        + "::context"
    )

    rng = random.Random(
        numeric_seed
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

    ground_truth = (
        enrich_ground_truth_explanation(
            task_type=task_type,
            scenario_id=scenario_id,
            raw_record=raw_record,
            ground_truth=ground_truth,
        )
    )

    validate_ground_truth_or_raise(
        ground_truth
    )

    if (
        task_type
        == "experiment_diagnosis"
        and ground_truth[
            "primary_issue"
        ]
        != expected_issue
    ):
        raise ValueError(
            f"{template_id} variant "
            f"{variant_index}: expected "
            f"{expected_issue}, got "
            f"{ground_truth['primary_issue']}"
        )

    if (
        task_type
        != "experiment_diagnosis"
        and ground_truth[
            "primary_issue"
        ]
        != "not_applicable"
    ):
        raise ValueError(
            f"{template_id}: "
            "non-diagnosis primary_issue "
            "must be not_applicable"
        )

    sample_id = (
        "full_"
        + hashlib.sha1(
            variant_key.encode(
                "utf-8"
            )
        ).hexdigest()[:14]
    )

    context_rng = random.Random(
        context_seed
    )

    run_context = build_run_context(
        sample_id,
        context_rng,
    )

    core_prompt = render_prompt_v2(
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

    prompt = (
        format_run_context(
            run_context
        )
        + "\n\n"
        + core_prompt
    )

    assistant_target = (
        json.dumps(
            ground_truth,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return {
        "sample_id":
            sample_id,

        "dataset_stage":
            "standard_full",

        "split":
            quota["split"],

        "task_type":
            task_type,

        "primary_issue":
            ground_truth[
                "primary_issue"
            ],

        "scenario_family_id":
            scenario_id,

        "template_family_id":
            template_id,

        "presentation_family_id":
            quota[
                "presentation_family_id"
            ],

        "render_mode":
            template[
                "render_mode"
            ],

        "run_context":
            run_context,

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
            "generator_version":
                "full_v1",

            "variant_index":
                variant_index,

            "deterministic_seed":
                numeric_seed,

            "context_seed":
                context_seed,
        },
    }


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

    samples = []

    for quota in sampling_plan[
        "template_quotas"
    ]:
        template_id = quota[
            "template_family_id"
        ]

        sample_count = quota[
            "full_sample_count"
        ]

        template = template_map[
            template_id
        ]

        for variant_index in range(
            sample_count
        ):
            samples.append(
                build_full_sample(
                    quota=quota,
                    template=template,
                    variant_index=variant_index,
                )
            )

    samples.sort(
        key=lambda item: (
            item["split"],
            item["task_type"],
            item["scenario_family_id"],
            item["template_family_id"],
            item["sample_id"],
        )
    )

    write_jsonl(
        OUTPUT_PATH,
        samples,
    )

    for split_name, split_path in (
        SPLIT_PATHS.items()
    ):
        split_samples = [
            sample
            for sample in samples
            if sample[
                "split"
            ]
            == split_name
        ]

        write_jsonl(
            split_path,
            split_samples,
        )

    split_counter = Counter(
        sample[
            "split"
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
            "primary_issue"
        ]
        for sample in samples
        if sample[
            "task_type"
        ]
        == "experiment_diagnosis"
    )

    severity_counter = Counter(
        sample[
            "ground_truth"
        ][
            "severity"
        ]
        for sample in samples
    )

    stats = {
        "total_samples":
            len(samples),

        "split_distribution":
            dict(split_counter),

        "task_type_distribution":
            dict(task_counter),

        "diagnosis_issue_distribution":
            dict(issue_counter),

        "severity_distribution":
            dict(severity_counter),

        "unique_template_families":
            len(
                {
                    sample[
                        "template_family_id"
                    ]
                    for sample in samples
                }
            ),

        "unique_scenario_families":
            len(
                {
                    sample[
                        "scenario_family_id"
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
        "FULL DATASET GENERATION PASSED"
    )

    print(
        "Master JSONL（主数据文件）：",
        OUTPUT_PATH,
    )

    print(
        "Full Samples（正式样本）：",
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

    print(
        "Diagnosis Issue Distribution"
        "（诊断主要问题分布）：",
        dict(issue_counter),
    )

    print(
        "Severity Distribution（严重程度分布）：",
        dict(severity_counter),
    )


if __name__ == "__main__":
    main()
