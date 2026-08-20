import json
from pathlib import Path


ROOT = Path.cwd()

SCENARIO_PATH = ROOT / "configs" / "scenario_families_v1.json"
OUTPUT_PATH = ROOT / "configs" / "template_families_v1.json"


PRESENTATION_FAMILIES = [
    {
        "presentation_family_id": "PF_STRUCTURED_BLOCK",
        "split": "train",
        "render_mode": "structured_block",
        "description": (
            "结构化字段块。使用稳定字段标签逐行呈现实验信息，"
            "但不使用 JSON 作为用户输入格式。"
        ),
        "render_contract": (
            "Render all required observations as labeled key-value "
            "lines under a compact experiment snapshot."
        ),
    },
    {
        "presentation_family_id": "PF_TABULAR_REPORT",
        "split": "train",
        "render_mode": "tabular_report",
        "description": (
            "表格报告。将必要实验指标组织成两列或多列表格。"
        ),
        "render_contract": (
            "Render required observations in a compact table-like "
            "report with explicit field names and values."
        ),
    },
    {
        "presentation_family_id": "PF_TRACKER_EXPORT",
        "split": "train",
        "render_mode": "tracker_export",
        "description": (
            "实验跟踪器导出格式。模拟 experiment tracker "
            "中的 run summary 和 key=value 字段。"
        ),
        "render_contract": (
            "Render observations as an experiment-tracker style "
            "run export using explicit key=value fields."
        ),
    },
    {
        "presentation_family_id": "PF_CONCISE_NOTE",
        "split": "validation",
        "render_mode": "concise_note",
        "description": (
            "简洁实验记录。将必要指标写成短篇自然语言记录。"
        ),
        "render_contract": (
            "Render every required observation inside a concise "
            "natural-language experiment note."
        ),
    },
    {
        "presentation_family_id": "PF_DEBUG_TICKET",
        "split": "test",
        "render_mode": "debug_ticket",
        "description": (
            "调试工单格式。使用 Context、Observations、Question "
            "等结构描述实验问题。"
        ),
        "render_contract": (
            "Render the sample as a debugging ticket with explicit "
            "Context, Observations, and Question sections."
        ),
    },
    {
        "presentation_family_id": "PF_NARRATIVE_SUMMARY",
        "split": "test",
        "render_mode": "narrative_summary",
        "description": (
            "自然语言实验总结。将指标嵌入连续的实验运行描述中。"
        ),
        "render_contract": (
            "Render all required observations inside a continuous "
            "narrative run summary without a key-value layout."
        ),
    },
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if not SCENARIO_PATH.exists():
        raise SystemExit(
            f"BUILD FAILED: missing {SCENARIO_PATH}"
        )

    scenario_config = load_json(SCENARIO_PATH)

    template_families = []

    for task_type, task_config in scenario_config[
        "task_types"
    ].items():
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

            required_inputs = scenario.get(
                "required_inputs",
                [],
            )

            for presentation in PRESENTATION_FAMILIES:
                presentation_id = presentation[
                    "presentation_family_id"
                ]

                template_id = (
                    f"{scenario_id}__{presentation_id}"
                )

                template_families.append(
                    {
                        "template_family_id": template_id,
                        "task_type": task_type,
                        "primary_issue": primary_issue,
                        "scenario_family_id": scenario_id,
                        "presentation_family_id":
                            presentation_id,
                        "split": presentation["split"],
                        "status": "ready",
                        "required_inputs": required_inputs,
                        "render_mode":
                            presentation["render_mode"],
                        "render_contract":
                            presentation["render_contract"],
                        "variant_policy": {
                            "numeric_variants_allowed": True,
                            "paraphrase_variants_allowed": True,
                            "variants_must_inherit_split": True,
                            "surface_paraphrase_does_not_create_new_family":
                                True,
                        },
                    }
                )

    output = {
        "version": "1.0",
        "description": (
            "Template Family configuration for "
            "DeepLearning-Copilot Standard Set."
        ),
        "design_policy": {
            "standard_split_unit": "template_family_id",
            "split_assignment_before_generation": True,
            "random_split_after_generation": False,
            "scenario_family_may_span_standard_splits": True,
            "unseen_scenario_family_reserved_for_hard_test": True,
            "presentation_family_split_is_global": True,
        },
        "presentation_families": PRESENTATION_FAMILIES,
        "template_families": template_families,
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

    split_counts = {}

    for template in template_families:
        split_name = template["split"]

        split_counts[split_name] = (
            split_counts.get(split_name, 0)
            + 1
        )

    print("TEMPLATE FAMILY BUILD PASSED")
    print("输出文件：", OUTPUT_PATH)
    print(
        "presentation families：",
        len(PRESENTATION_FAMILIES),
    )
    print(
        "template families：",
        len(template_families),
    )
    print(
        "split family counts：",
        split_counts,
    )


if __name__ == "__main__":
    main()
