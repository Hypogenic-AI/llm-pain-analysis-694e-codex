# Downloaded Datasets

This directory contains datasets for LLM refusal/avoidance analysis. Full dataset artifacts are local-only and excluded from git by `.gitignore`.

Generated: 2026-03-22T06:37:17.350830+00:00

## Dataset: hh_rlhf

- **Source**: `Anthropic/hh-rlhf`
- **Task fit**: helpful/harmless preference and refusal behavior
- **Location (local)**: `datasets/hh_rlhf/hf_disk`
- **Splits**: train (160800), test (8552)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf")
dataset.save_to_disk("datasets/hh_rlhf/hf_disk")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/hh_rlhf/hf_disk")
```

### Sample Data
Sample files: `datasets/hh_rlhf/samples/*_sample.json`

### Notes
- Validated by loading each split and exporting 10 examples/split.
- Feature schemas are recorded in `datasets/dataset_summary.json`.

## Dataset: toxic_chat

- **Source**: `lmsys/toxic-chat:toxicchat0124`
- **Task fit**: toxic prompt safety classification and refusal risk
- **Location (local)**: `datasets/toxic_chat/hf_disk`
- **Splits**: train (5082), test (5083)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
dataset.save_to_disk("datasets/toxic_chat/hf_disk")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/toxic_chat/hf_disk")
```

### Sample Data
Sample files: `datasets/toxic_chat/samples/*_sample.json`

### Notes
- Validated by loading each split and exporting 10 examples/split.
- Feature schemas are recorded in `datasets/dataset_summary.json`.

## Dataset: truthful_qa

- **Source**: `truthful_qa:generation`
- **Task fit**: truthfulness vs refusal/avoidance tradeoff
- **Location (local)**: `datasets/truthful_qa/hf_disk`
- **Splits**: validation (817)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("truthful_qa", "generation")
dataset.save_to_disk("datasets/truthful_qa/hf_disk")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthful_qa/hf_disk")
```

### Sample Data
Sample files: `datasets/truthful_qa/samples/*_sample.json`

### Notes
- Validated by loading each split and exporting 10 examples/split.
- Feature schemas are recorded in `datasets/dataset_summary.json`.
