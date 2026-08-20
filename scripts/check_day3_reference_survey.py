from pathlib import Path


ROOT = Path.cwd()
SURVEY_PATH = ROOT / "docs" / "DAY3_TEMPLATE_REFERENCE_SURVEY.md"


def main() -> None:
    errors = []

    if not SURVEY_PATH.exists():
        errors.append(
            f"Reference Survey 文件不存在：{SURVEY_PATH}"
        )

    if errors:
        print("DAY 3 REFERENCE SURVEY 验证失败")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    text = SURVEY_PATH.read_text(encoding="utf-8")

    required_sections = [
        "# Day 3 Template Reference Survey",
        "## 2. 当前覆盖范围与待补任务",
        "## 3. License 与引用政策",
        "## 4. 阈值解释政策",
        "# 5. Deepchecks 调研",
        "# 6. Cleanlab 调研",
        "# 7. scikit-learn 调研",
        "# 8. imbalanced-learn 调研",
        "# 9. Keras 调研",
        "# 10. Proposed Scenario Taxonomy",
        "# 11. Template-Family 层级",
        "# 12. Standard Split Policy",
        "# 13. Dataset 政策",
        "# 14. 与 Day 2 组件的兼容性",
        "# 15. 结论",
    ]

    required_projects = [
        "Deepchecks",
        "Cleanlab",
        "scikit-learn",
        "imbalanced-learn",
        "Keras",
    ]

    required_licenses = [
        "AGPL-3.0",
        "Apache-2.0",
        "BSD-3-Clause",
        "MIT",
    ]

    required_phrases = [
        "LLM must not assign labels",
        "No third-party implementation code",
        "Standard split unit",
        "template_family_id",
        (
            "scenario_family can span multiple Standard Splits "
            "through different template_family_id values"
        ),
        "completely unseen `scenario_family`",
        "禁止生成 Dataset 后再执行 random split",
        "experiment_diagnosis",
        "metric_interpretation",
        "model_comparison",
        "最终 Dataset 不能只有 `experiment_diagnosis`",
    ]

    required_domains = [
        "docs.deepchecks.com",
        "github.com/deepchecks/deepchecks",
        "docs.cleanlab.ai",
        "github.com/cleanlab/cleanlab",
        "scikit-learn.org",
        "github.com/scikit-learn/scikit-learn",
        "imbalanced-learn.org",
        "github.com/scikit-learn-contrib/imbalanced-learn",
        "keras.io",
        "github.com/keras-team/keras",
    ]

    for section in required_sections:
        if section not in text:
            errors.append(f"缺少必要章节：{section}")

    for project in required_projects:
        if project not in text:
            errors.append(f"缺少必要项目：{project}")

    for license_name in required_licenses:
        if license_name not in text:
            errors.append(f"缺少 License：{license_name}")

    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"缺少必要内容：{phrase}")

    for domain in required_domains:
        if domain not in text:
            errors.append(f"缺少官方来源：{domain}")

    license_table_header = (
        "| Project | License | Usage policy |\n"
        "| --- | --- | --- |"
    )

    if license_table_header not in text:
        errors.append("License 表格表头格式错误")

    scenario_taxonomy_header = (
        "| Issue | Proposed scenario_family | "
        "Current compatibility | Main source |\n"
        "| --- | --- | --- | --- |"
    )

    if scenario_taxonomy_header not in text:
        errors.append("Scenario Taxonomy 表格表头格式错误")

    forbidden_phrases = [
        "DAY3\\_TEMPLATE\\_REFERENCE\\_SURVEY.md",
        "required\\_sections",
        "read\\_text",
    ]

    for phrase in forbidden_phrases:
        if phrase in text:
            errors.append(f"发现错误或过期内容：{phrase}")

    if text.count("AGPL-3.0") < 2:
        errors.append("Deepchecks 的 AGPL-3.0 政策记录不足")

    if "must not copy Deepchecks implementation code" not in text:
        errors.append("缺少禁止复制 Deepchecks 实现代码的说明")

    if (
        "不要求整个 `scenario_family` 只能属于一个 "
        "Standard Split"
        not in text
    ):
        errors.append(
            "缺少 scenario_family 可以跨 split 的中文说明"
        )

    if errors:
        print("DAY 3 REFERENCE SURVEY 验证失败")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    print("DAY3 REFERENCE SURVEY VALIDATION PASSED")
    print("文件：", SURVEY_PATH)
    print("字符数：", len(text))
    print("必要章节数：", len(required_sections))
    print("已检查项目数：", len(required_projects))
    print("已检查 License 数：", len(required_licenses))


if __name__ == "__main__":
    main()
