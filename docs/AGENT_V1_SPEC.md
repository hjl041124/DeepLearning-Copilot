# DeepLearning-Copilot Agent v1 Specification


# 1. Project Context（项目背景）


DeepLearning-Copilot is an intelligent deep learning experiment diagnosis system.


The existing project already contains:


- Dataset Construction（数据集构建）
- Scenario Taxonomy（场景分类体系）
- Template Family Design（模板族设计）
- Feature Calculation（特征计算）
- Rule Engine（规则引擎）
- Ground Truth Builder（标准答案生成器）
- Evaluation Framework（评估框架）
- QLoRA Fine-tuned Diagnosis Model（QLoRA微调诊断模型）


The Agent layer is an extension of the existing system.


The Agent MUST NOT replace existing modules.


The Agent should provide workflow orchestration and tool integration on top of the existing diagnosis system.


---


# 2. Agent Goal（智能体目标）


The goal of Agent v1 is:


Build an experiment diagnosis assistant that can:


1. Receive experiment information（接收实验信息）


2. Collect structured analysis evidence（收集结构化分析证据）


3. Call analysis tools（调用分析工具）


4. Construct diagnosis context（构建诊断上下文）


5. Invoke the final QLoRA diagnosis model（调用最终QLoRA诊断模型）


6. Generate an interpretable diagnosis report（生成可解释诊断报告）


---


# 3. Agent Responsibility（智能体职责）


The Agent is responsible for:


- Workflow orchestration（工作流编排）

- State management（状态管理）

- Tool calling（工具调用）

- Context construction（上下文构建）

- Diagnosis report generation（诊断报告生成）


The Agent is NOT responsible for:


- Training models（训练模型）

- Creating diagnosis rules（创建诊断规则）

- Generating Ground Truth（生成标准答案）

- Replacing Rule Engine（替代规则引擎）

- Rebuilding Evaluation Framework（重建评估框架）


---


# 4. Design Principle（设计原则）


DeepLearning-Copilot follows:


Deterministic Analysis + LLM Reasoning


The existing deterministic modules provide:


- calculated features（计算特征）

- structured evidence（结构化证据）

- evaluation standards（评估标准）


The QLoRA model provides:


- diagnosis reasoning（诊断推理）

- evidence selection（证据选择）

- recommendation generation（建议生成）


The Agent provides:


- workflow control（流程控制）

- information coordination（信息协调）




---

# 5. Agent Architecture（智能体架构）


Agent v1 uses a controlled workflow architecture.


The system contains four main layers.


## 5.1 Workflow Layer（工作流层）


Responsible for:


- workflow execution（工作流执行）

- state management（状态管理）

- node coordination（节点协调）

- execution control（执行控制）


The Workflow Layer should use LangGraph.


The workflow should remain controlled and reproducible.



## 5.2 Tool Layer（工具层）


Responsible for collecting structured experiment information.


Tools include:


- Metric Analysis Tool（指标分析工具）

- Training Log Analyzer（训练日志分析工具）

- Dataset Checker（数据检查工具）


Tools should provide structured features for diagnosis.



## 5.3 Reasoning Layer（推理层）


Responsible for:


- calling the Final QLoRA Diagnosis Model

- diagnosis reasoning

- evidence selection

- recommendation generation



The reasoning model is:


Qwen3-4B-Instruct-2507

+

DeepLearning-Copilot Final QLoRA Adapter



## 5.4 Storage Layer（存储层）


Responsible for:


- experiment history storage

- diagnosis record storage

- execution record storage



Agent v1 uses SQLite storage.



---


# 6. Workflow Sequence（工作流程）


The default Agent execution sequence:


## Step 1: Receive Experiment Input（接收实验输入）


The Agent receives:


- experiment information

- metrics

- training logs

- dataset information



## Step 2: Parse Experiment Context（解析实验上下文）


The Agent extracts:


- model information

- dataset information

- available experiment signals



## Step 3: Execute Analysis Tools（执行分析工具）


The Agent executes:


- Metric Analysis Tool

- Training Log Analyzer

- Dataset Checker



## Step 4: Build Diagnosis Context（构建诊断上下文）


The Agent combines:


- experiment context

- metric features

- log features

- dataset features



## Step 5: Invoke QLoRA Diagnosis Model（调用QLoRA诊断模型）


The combined context is sent to:


DeepLearning-Copilot Final QLoRA Model



## Step 6: Generate Report（生成报告）


The Agent generates:


- primary issue

- severity

- evidence codes

- recommended action codes

- explanation



## Step 7: Store Experiment History（保存实验历史）


The execution result is stored for future analysis.



---


# 7. Agent State Schema（状态结构）


The Agent state stores information required during one diagnosis workflow.


Recommended fields:


- experiment_id

- user_input

- experiment_context

- metric_features

- log_features

- dataset_features

- combined_context

- diagnosis

- report



## experiment_id


Unique identifier for each experiment.



## user_input


Original user request or experiment description.



## experiment_context


Contains original experiment information:


- model name

- dataset information

- training configuration

- available metrics



## metric_features


Structured features extracted from experiment metrics.



Examples:


- accuracy_macro_gap

- class_performance_gap

- generalization_gap



## log_features


Structured features extracted from training logs.



Examples:


- train_validation_gap

- late_validation_degradation

- nan_detected



## dataset_features


Structured dataset information.



Examples:


- class_distribution

- imbalance_ratio



## combined_context


Final structured context provided to the QLoRA model.



## diagnosis


QLoRA model output.



## report


Human-readable final diagnosis report.





---

# 14. QLoRA Model Integration（QLoRA模型集成）


The Agent uses the final DeepLearning-Copilot QLoRA model for diagnosis reasoning.


The Agent does NOT perform diagnosis reasoning itself.

The Agent prepares structured context and sends it to the QLoRA model.



---


# 15. Final Model Information（最终模型信息）


Base Model（基础模型）:


Qwen3-4B-Instruct-2507



Fine-tuning Method（微调方法）:


QLoRA



Final Adapter Path（最终Adapter路径）:


models/qwen3-4b-dlcopilot-final-qlora



The model wrapper should load this adapter for inference.



---


# 16. QLoRA Input Schema（模型输入结构）


The Agent should provide structured diagnosis context.


The input should include:


- metric_features

- log_features

- dataset_features

- experiment_context



Example:


{
    "metric_features": {},

    "log_features": {},

    "dataset_features": {},

    "experiment_context": {}
}



---


# 17. QLoRA Output Schema（模型输出结构）


The QLoRA model should return structured diagnosis information.


Required fields:


- primary_issue

- severity

- evidence_codes

- recommended_action_codes

- explanation



Example:


{
    "primary_issue": "",

    "severity": "",

    "evidence_codes": [],

    "recommended_action_codes": [],

    "explanation": ""
}



---


# 18. Model Responsibility Boundary（模型职责边界）


QLoRA Model is responsible for:


- diagnosis reasoning（诊断推理）

- evidence selection（证据选择）

- recommendation generation（建议生成）


QLoRA Model is NOT responsible for:


- collecting raw experiment data

- calculating deterministic features

- executing tools

- managing workflow state





---

# 19. Storage Design（存储设计）


Agent v1 uses SQLite for experiment history storage.



The storage is used for:


- experiment records（实验记录）

- tool execution results（工具执行结果）

- diagnosis results（诊断结果）

- execution timestamps（执行时间）



Recommended information:


- experiment_id

- input_context

- tool_results

- diagnosis

- report

- timestamp



Agent v1 does NOT include:


- Vector Database（向量数据库）

- RAG（检索增强生成）

- Long-term semantic memory（长期语义记忆）



These can be considered future extensions.



---


# 20. Development Constraints（开发限制）


The implementation must respect the existing DeepLearning-Copilot system.


Codex MUST:


- Read the existing repository before implementation.

- Understand existing modules before adding code.

- Keep current Dataset Pipeline unchanged.

- Keep current Rule Engine unchanged.

- Keep current Ground Truth Builder unchanged.

- Keep current Evaluation Framework unchanged.

- Keep QLoRA training pipeline unchanged.



The Agent implementation should add new modules only.



Recommended new modules:


agent/

tools/

model/

storage/



The exact location should follow the current repository structure.



---


# 21. Forbidden Changes（禁止修改）


The Agent implementation MUST NOT:


- rewrite dataset generation pipeline

- create new diagnosis rules

- duplicate Rule Engine logic

- replace Ground Truth generation

- modify QLoRA training code

- introduce Multi-Agent architecture

- introduce unnecessary external APIs



---


# 22. Acceptance Criteria（验收标准）


Agent v1 is considered complete when the following requirements are satisfied.



## Workflow Requirement（工作流要求）


The Agent can execute:


Input

↓

Tool execution

↓

Context construction

↓

QLoRA diagnosis

↓

Report generation



---


## Tool Requirement（工具要求）


The Agent can successfully integrate:


- Metric Analysis Tool

- Training Log Analyzer

- Dataset Checker



---


## Model Requirement（模型要求）


The Agent can load:


models/qwen3-4b-dlcopilot-final-qlora



The Agent can send structured context to the model and receive diagnosis output.



---


## Output Requirement（输出要求）


The final report should contain:


- primary_issue

- severity

- evidence_codes

- recommended_action_codes

- explanation



---


# 23. Future Extensions（未来扩展）


Possible future improvements:


- MLflow integration

- WandB integration

- Experiment retrieval

- Vector memory

- External service integration



These features are NOT included in Agent v1.



