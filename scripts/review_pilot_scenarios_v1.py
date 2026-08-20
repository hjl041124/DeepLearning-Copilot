import json
from collections import defaultdict
from pathlib import Path


ROOT = Path.cwd()

DATA_PATH = (
    ROOT
    / "data"
    / "generated"
    / "pilot_dataset_v1.jsonl"
)


SCENARIO_NAMES = {
    "ED_OF_GENERALIZATION_GAP_LATE_DEGRADATION":
        "泛化差距明显 + 后期验证性能退化",
    "ED_OF_VALIDATION_LOSS_RISES":
        "验证损失先下降后重新上升",
    "ED_UF_LOW_SCORES_VS_REFERENCE":
        "训练和验证性能均明显低于参考水平",
    "ED_UF_REFERENCE_SHORTFALL_PLATEAU":
        "参考性能不足 + 训练进入平台期",
    "ED_OP_STRONG_LOSS_OSCILLATION":
        "训练损失强烈震荡",
    "ED_OP_NAN_OR_INF":
        "训练出现 NaN / Inf",
    "ED_CI_SKEW_WITH_CLASSWISE_COLLAPSE":
        "类别数量偏斜并伴随逐类别性能坍塌",
    "ED_CI_ACCURACY_MASKS_MACRO_F1":
        "高准确率掩盖较低的宏平均 F1",
    "ED_CI_MINORITY_PERFORMANCE_COLLAPSE":
        "少数类样本不足且性能明显坍塌",
    "ED_DQ_HIGH_LABEL_NOISE":
        "标签噪声率过高",
    "ED_DQ_HIGH_DUPLICATE_RATE":
        "重复样本率过高",
    "ED_DQ_SPLIT_OVERLAP":
        "训练与验证划分存在样本重叠",
    "ED_DQ_PREPROCESSING_MISMATCH":
        "不同数据划分的预处理不一致",
    "ED_DQ_EXPLICIT_DISTRIBUTION_SHIFT":
        "明确存在数据分布偏移",
    "ED_NC_STABLE_CONVERGENCE_SMALL_GAP":
        "稳定收敛且训练验证差距较小",
    "ED_NC_BALANCED_CLASSWISE_PERFORMANCE":
        "类别均衡且逐类别性能接近",
    "MI_ACCURACY_MACRO_F1_GAP":
        "解读准确率与宏平均 F1 的明显差距",
    "MI_PRECISION_RECALL_TRADEOFF":
        "解读精确率与召回率权衡",
    "MI_CLASSWISE_PERFORMANCE_GAP":
        "解读不同类别之间的性能差异",
    "MI_TRAIN_VALIDATION_GAP":
        "解读训练与验证指标差距",
    "MC_CLEAR_QUALITY_WINNER":
        "两个模型存在明确质量优胜者",
    "MC_QUALITY_EFFICIENCY_TRADEOFF":
        "模型质量与推理效率之间的权衡",
    "MC_IMBALANCED_METRIC_COMPARISON":
        "类别不平衡条件下的模型指标比较",
    "MC_NO_CLEAR_WINNER":
        "两个模型没有明显优胜者",
}


PRESENTATION_ORDER = [
    "PF_STRUCTURED_BLOCK",
    "PF_CONCISE_NOTE",
    "PF_DEBUG_TICKET",
    "PF_TABULAR_REPORT",
    "PF_NARRATIVE_SUMMARY",
    "PF_TRACKER_EXPORT",
]


def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def main():
    records = load_jsonl(
        DATA_PATH
    )

    grouped = defaultdict(
        list
    )

    for record in records:
        grouped[
            record[
                "scenario_family_id"
            ]
        ].append(
            record
        )

    scenario_ids = sorted(
        grouped
    )

    for index, scenario_id in enumerate(
        scenario_ids
    ):
        preferred = PRESENTATION_ORDER[
            index
            % len(PRESENTATION_ORDER)
        ]

        candidates = grouped[
            scenario_id
        ]

        selected = next(
            (
                item
                for item in candidates
                if item[
                    "presentation_family_id"
                ]
                == preferred
            ),
            candidates[0],
        )

        print()
        print(
            "=" * 90
        )

        print(
            f"SCENARIO {index + 1}"
        )

        print(
            "Scenario Family（场景族）：",
            scenario_id,
        )

        print(
            "中文含义：",
            SCENARIO_NAMES.get(
                scenario_id,
                "未定义",
            ),
        )

        print(
            "Presentation Family（呈现结构族）：",
            selected[
                "presentation_family_id"
            ],
        )

        print(
            "Split（数据划分）：",
            selected[
                "split"
            ],
        )

        print(
            "Task Type（任务类型）：",
            selected[
                "task_type"
            ],
        )

        print()
        print(
            "--- PROMPT（提示词） ---"
        )
        print(
            selected[
                "prompt"
            ]
        )

        print()
        print(
            "--- GROUND TRUTH（标准答案） ---"
        )
        print(
            json.dumps(
                selected[
                    "ground_truth"
                ],
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
