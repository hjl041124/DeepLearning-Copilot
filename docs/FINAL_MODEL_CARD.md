# DeepLearning-Copilot Final QLoRA Model

## Model

Base Model:

Qwen3-4B-Instruct-2507

Fine-tuning Method:

QLoRA (4-bit NF4)

Adapter:

LoRA

## Training Evolution

### QLoRA v1

Goal:

Schema Alignment

- task vocabulary
- output format
- evidence codes
- action codes


### QLoRA v2

Goal:

Hard Reasoning Improvement

- boundary reasoning
- invariance reasoning
- evidence grounding


### QLoRA v3 (Final)

Goal:

Advanced Hard Reasoning

- priority composition
- directional boundary
- output completion


## Final Model Directory

models/qwen3-4b-dlcopilot-final-qlora


## Evaluation

Datasets:

- Standard Test
- Hard Test


Metrics:

- Task Type Valid Rate
- Primary Issue Accuracy
- Evidence Exact Match
- Recommendation Exact Match
- Core Exact Match
