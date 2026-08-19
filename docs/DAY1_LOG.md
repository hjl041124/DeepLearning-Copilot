# Day 1 Experiment Log（第一天實驗記錄）

## Project（專案）

DeepLearning-Copilot

## Model（模型）

Qwen/Qwen3-4B-Instruct-2507

## Day 1 Goal（第一天目標）

Environment Setup（環境配置）
Model Download（模型下載）
Base Model Smoke Test（基線模型最小推理測試）

## Hardware（硬體）

- GPU: NVIDIA GeForce RTX 4090 D
- VRAM: 24564 MiB
- NVIDIA Driver: 560.35.03
- Driver CUDA Support: 12.6
- CUDA Toolkit: 12.4.131
- Data Disk: 150GB

## Project Environment（專案環境）

- Python: 3.11.15
- PyTorch: 2.13.0+cu126
- PyTorch CUDA: 12.6
- Transformers: 5.15.0
- Accelerate: 1.14.0
- PEFT: 0.20.0
- TRL: 1.9.1
- bitsandbytes: 0.50.0
- Datasets: 5.0.1

## Base Model Smoke Test（基線模型最小推理測試）

Experiment ID:
DLC-D1-E02

Prompt:
In one or two sentences, explain what overfitting means in a deep learning experiment.

Result:
PASSED

Observed Runtime（實際運行結果）:

- Model load time: 3.96 seconds
- GPU memory allocated after loading: 7.493 GB
- Input tokens: 26
- Generation time: 3.16 seconds
- Peak GPU memory during generation: 7.51 GB

Model Response（模型回答）:

Overfitting in a deep learning experiment occurs when a model learns the training data too well, capturing noise and irrelevant patterns instead of the underlying relationships, which causes it to perform poorly on new, unseen data.

## Important Notes（重要說明）

- No training was performed on Day 1.
- No dataset was constructed on Day 1.
- Smoke-test runtime is not treated as a formal benchmark.
- Smoke-test GPU memory is not treated as QLoRA training memory.
- No experimental metric has been fabricated.

## Day 1 Status（第一天狀態）

COMPLETED
