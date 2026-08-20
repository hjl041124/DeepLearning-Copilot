import json
from pathlib import Path

from hypothesis import (
    given,
    settings,
    strategies as st,
)

from src.evaluation.hard_test_property_engine import (
    BOUNDARY_FAMILIES,
    INVARIANCE_FAMILIES,
    PRIORITY_FAMILIES,
    SUPPORTED_HARD_TEST_FAMILIES,
    build_class_imbalance_boundary_pair,
    build_deterministic_ground_truth,
    build_generalization_boundary_pair,
    build_invariance_pair,
    build_label_noise_boundary_pair,
    build_priority_composition,
    compute_diagnosis_features,
    decision_signature,
    get_threshold,
)


ROOT = Path.cwd()

SPEC_PATH = (
    ROOT
    / "configs"
    / "hard_test_spec_v1.json"
)


EPSILON = st.sampled_from(
    [
        0.001,
        0.003,
        0.005,
        0.01,
    ]
)


COMMON_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
)


def load_spec_family_ids():
    spec = json.loads(
        SPEC_PATH.read_text(
            encoding="utf-8"
        )
    )

    ids = set()

    for group in spec[
        "hard_test_groups"
    ]:
        for family in group[
            "families"
        ]:
            ids.add(
                family[
                    "hard_test_family_id"
                ]
            )

    return ids


def test_spec_coverage():
    spec_ids = (
        load_spec_family_ids()
    )

    assert (
        spec_ids
        == SUPPORTED_HARD_TEST_FAMILIES
    ), (
        "Property Engine（属性引擎）"
        "与 Hard Test Spec（困难测试规范）"
        "覆盖不一致"
    )

    assert len(
        BOUNDARY_FAMILIES
    ) == 3

    assert len(
        INVARIANCE_FAMILIES
    ) == 3

    assert len(
        PRIORITY_FAMILIES
    ) == 6


@COMMON_SETTINGS
@given(
    epsilon=EPSILON
)
def test_directional_generalization_gap(
    epsilon,
):
    pair = (
        build_generalization_boundary_pair(
            epsilon
        )
    )

    threshold = pair[
        "threshold"
    ]

    below_features = (
        compute_diagnosis_features(
            pair["below"]
        )
    )

    above_features = (
        compute_diagnosis_features(
            pair["above"]
        )
    )

    assert (
        below_features[
            "relative_generalization_gap"
        ]
        < threshold
    )

    assert (
        above_features[
            "relative_generalization_gap"
        ]
        > threshold
    )

    below_gt = (
        build_deterministic_ground_truth(
            pair["below"]
        )
    )

    above_gt = (
        build_deterministic_ground_truth(
            pair["above"]
        )
    )

    assert (
        below_gt[
            "primary_issue"
        ]
        == "no_clear_issue"
    )

    assert (
        above_gt[
            "primary_issue"
        ]
        == "overfitting"
    )


@COMMON_SETTINGS
@given(
    epsilon=EPSILON
)
def test_directional_class_imbalance_ratio(
    epsilon,
):
    pair = (
        build_class_imbalance_boundary_pair(
            epsilon
        )
    )

    threshold = pair[
        "threshold"
    ]

    below_features = (
        compute_diagnosis_features(
            pair["below"]
        )
    )

    above_features = (
        compute_diagnosis_features(
            pair["above"]
        )
    )

    assert (
        below_features[
            "class_imbalance_ratio"
        ]
        <= threshold
    )

    assert (
        above_features[
            "class_imbalance_ratio"
        ]
        > threshold
    )

    performance_threshold = (
        get_threshold(
            "class_performance_gap",
            "strong_min",
        )
    )

    assert (
        below_features[
            "class_performance_gap"
        ]
        >= performance_threshold
    )

    assert (
        above_features[
            "class_performance_gap"
        ]
        >= performance_threshold
    )

    below_gt = (
        build_deterministic_ground_truth(
            pair["below"]
        )
    )

    above_gt = (
        build_deterministic_ground_truth(
            pair["above"]
        )
    )

    assert (
        below_gt[
            "primary_issue"
        ]
        == "class_imbalance"
    )

    assert (
        above_gt[
            "primary_issue"
        ]
        == "no_clear_issue"
    )


@COMMON_SETTINGS
@given(
    epsilon=EPSILON
)
def test_directional_label_noise(
    epsilon,
):
    pair = (
        build_label_noise_boundary_pair(
            epsilon
        )
    )

    threshold = pair[
        "threshold"
    ]

    below_features = (
        compute_diagnosis_features(
            pair["below"]
        )
    )

    above_features = (
        compute_diagnosis_features(
            pair["above"]
        )
    )

    assert (
        below_features[
            "label_noise_rate"
        ]
        < threshold
    )

    assert (
        above_features[
            "label_noise_rate"
        ]
        > threshold
    )

    below_gt = (
        build_deterministic_ground_truth(
            pair["below"]
        )
    )

    above_gt = (
        build_deterministic_ground_truth(
            pair["above"]
        )
    )

    assert (
        below_gt[
            "primary_issue"
        ]
        == "no_clear_issue"
    )

    assert (
        above_gt[
            "primary_issue"
        ]
        == "data_quality_issue"
    )


METADATA_STRATEGY = (
    st.fixed_dictionaries(
        {
            "gpu_name":
                st.sampled_from(
                    [
                        "RTX 3090",
                        "RTX 4090",
                        "A100",
                        "L40S",
                    ]
                ),

            "seed":
                st.integers(
                    min_value=0,
                    max_value=99999,
                ),

            "run_name":
                st.text(
                    alphabet=
                        "abcdefghijklmnopqrstuvwxyz0123456789_",
                    min_size=3,
                    max_size=16,
                ),

            "batch_size":
                st.sampled_from(
                    [
                        16,
                        32,
                        64,
                        128,
                    ]
                ),

            "optimizer_name":
                st.sampled_from(
                    [
                        "AdamW",
                        "Adam",
                        "SGD",
                    ]
                ),
        }
    )
)


@COMMON_SETTINGS
@given(
    metadata=METADATA_STRATEGY
)
def test_invariance_diagnosis_metadata(
    metadata,
):
    for family_id in [
        "HT_INV_OVERFITTING_METADATA",
        "HT_INV_DATA_QUALITY_METADATA",
    ]:
        pair = (
            build_invariance_pair(
                family_id,
                metadata,
            )
        )

        base_gt = (
            build_deterministic_ground_truth(
                pair["base"]
            )
        )

        perturbed_gt = (
            build_deterministic_ground_truth(
                pair["perturbed"]
            )
        )

        assert (
            decision_signature(
                base_gt
            )
            == decision_signature(
                perturbed_gt
            )
        )


MODEL_METADATA_STRATEGY = (
    st.fixed_dictionaries(
        {
            "gpu_name":
                st.sampled_from(
                    [
                        "RTX 4090",
                        "A100",
                        "H100",
                    ]
                ),

            "seed":
                st.integers(
                    min_value=0,
                    max_value=99999,
                ),

            "training_epochs":
                st.integers(
                    min_value=1,
                    max_value=300,
                ),

            "framework_version":
                st.sampled_from(
                    [
                        "2.4",
                        "2.5",
                        "2.6",
                    ]
                ),

            "checkpoint_name":
                st.text(
                    alphabet=
                        "abcdefghijklmnopqrstuvwxyz0123456789_-",
                    min_size=3,
                    max_size=18,
                ),
        }
    )
)


@COMMON_SETTINGS
@given(
    metadata=MODEL_METADATA_STRATEGY
)
def test_invariance_model_comparison_metadata(
    metadata,
):
    pair = (
        build_invariance_pair(
            "HT_INV_MODEL_COMPARISON_METADATA",
            metadata,
        )
    )

    base_gt = (
        build_deterministic_ground_truth(
            pair["base"]
        )
    )

    perturbed_gt = (
        build_deterministic_ground_truth(
            pair["perturbed"]
        )
    )

    assert (
        decision_signature(
            base_gt
        )
        == decision_signature(
            perturbed_gt
        )
    )


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
)
@given(
    family_id=st.sampled_from(
        sorted(
            PRIORITY_FAMILIES
        )
    )
)
def test_priority_composition(
    family_id,
):
    case = (
        build_priority_composition(
            family_id
        )
    )

    # 每个 Component（组成问题）单独存在时，
    # 必须真的触发对应 Issue（问题）。
    for (
        expected_issue,
        component_record,
    ) in case[
        "components"
    ].items():
        component_gt = (
            build_deterministic_ground_truth(
                component_record
            )
        )

        assert (
            component_gt[
                "primary_issue"
            ]
            == expected_issue
        ), (
            family_id,
            expected_issue,
            component_gt,
        )

    composed_gt = (
        build_deterministic_ground_truth(
            case[
                "composed"
            ]
        )
    )

    assert (
        composed_gt[
            "primary_issue"
        ]
        == case[
            "expected_primary_issue"
        ]
    ), (
        family_id,
        composed_gt,
    )


def deterministic_family_audit():
    # Boundary（边界）使用固定 epsilon 再审一次
    for builder, below_issue, above_issue in [
        (
            build_generalization_boundary_pair,
            "no_clear_issue",
            "overfitting",
        ),
        (
            build_class_imbalance_boundary_pair,
            "class_imbalance",
            "no_clear_issue",
        ),
        (
            build_label_noise_boundary_pair,
            "no_clear_issue",
            "data_quality_issue",
        ),
    ]:
        pair = builder(
            0.005
        )

        below_gt = (
            build_deterministic_ground_truth(
                pair["below"]
            )
        )

        above_gt = (
            build_deterministic_ground_truth(
                pair["above"]
            )
        )

        assert (
            below_gt[
                "primary_issue"
            ]
            == below_issue
        )

        assert (
            above_gt[
                "primary_issue"
            ]
            == above_issue
        )

    # Priority（优先级）全部 6 个 Family（测试族）
    for family_id in sorted(
        PRIORITY_FAMILIES
    ):
        case = (
            build_priority_composition(
                family_id
            )
        )

        gt = (
            build_deterministic_ground_truth(
                case[
                    "composed"
                ]
            )
        )

        assert (
            gt[
                "primary_issue"
            ]
            == case[
                "expected_primary_issue"
            ]
        )


def main():
    test_spec_coverage()

    test_directional_generalization_gap()

    test_directional_class_imbalance_ratio()

    test_directional_label_noise()

    test_invariance_diagnosis_metadata()

    test_invariance_model_comparison_metadata()

    test_priority_composition()

    deterministic_family_audit()

    print(
        "DAY4 HARD TEST PROPERTY TESTS PASSED"
    )

    print(
        "Directional Boundary Families"
        "（方向性边界测试族）：",
        len(
            BOUNDARY_FAMILIES
        ),
    )

    print(
        "Invariance Families"
        "（不变性测试族）：",
        len(
            INVARIANCE_FAMILIES
        ),
    )

    print(
        "Priority Composition Families"
        "（优先级组合测试族）：",
        len(
            PRIORITY_FAMILIES
        ),
    )

    print(
        "Total Covered Families"
        "（总覆盖困难测试族）：",
        len(
            SUPPORTED_HARD_TEST_FAMILIES
        ),
    )


if __name__ == "__main__":
    main()
