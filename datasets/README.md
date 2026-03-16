# Downloaded Datasets

This directory contains datasets for the In-context If-Then Capacity research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: IFEval

### Overview
- **Source**: HuggingFace `google/IFEval`
- **Size**: 541 examples, ~115 KB
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Verifiable instruction following (25 constraint types)
- **Splits**: train (541)
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("google/IFEval")
dataset.save_to_disk("datasets/IFEval")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/IFEval")
```

### Notes
- Contains only unconditional constraints (format, length, keywords, etc.)
- Serves as baseline/control for measuring if-then instruction following gap
- Features: `key`, `prompt`, `instruction_id_list`, `kwargs`

---

## Dataset 2: ComplexBench

### Overview
- **Source**: https://github.com/thu-coai/ComplexBench
- **Size**: 1,150 instructions, ~6.5 MB
- **Format**: JSON file
- **Task**: Complex multi-constraint instruction following with composition types
- **License**: Research use

### Download Instructions

```bash
# From the ComplexBench GitHub repo
curl -L -o datasets/ComplexBench_data_final.json \
  "https://raw.githubusercontent.com/thu-coai/ComplexBench/main/data/data_final.json"
```

### Loading

```python
import json
with open("datasets/ComplexBench_data_final.json", "r") as f:
    data = json.load(f)
```

### Notes
- **CRITICAL**: Contains "Selection" composition type = explicit if-then-else conditional branching
- Also contains And, Chain, and nested compositions
- 4 constraint types (Lexical, Format, Semantic, Utility), 19 dimensions
- Primary in Chinese with English translations available
- Includes manually crafted yes/no scoring questions per constraint

---

## Dataset 3: SIFo (Sequential Instruction Following)

### Overview
- **Source**: https://github.com/shin-ee-chen/SIFo
- **Size**: 800 samples total, ~327 KB
- **Format**: JSONL files (4 files, one per task)
- **Task**: Causally dependent sequential instruction following
- **Splits**: 200 samples per task
- **License**: Research use

### Download Instructions

```bash
# From the SIFo GitHub repo
mkdir -p datasets/SIFo
for task in math qa security text_modification; do
  curl -L -o "datasets/SIFo/${task}.jsonl" \
    "https://raw.githubusercontent.com/shin-ee-chen/SIFo/main/data/${task}.jsonl"
done
```

### Loading

```python
import json
with open("datasets/SIFo/security.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
```

### Notes
- Security Rules task contains implicit if-then conditional reasoning
- Instructions are causally chained — each step depends on prior
- Verifiable without LLM judges
- 4 tasks: text_modification, qa, math, security

---

## Dataset 4: FollowBench

### Overview
- **Source**: HuggingFace `YuxinJiang/FollowBench`
- **Size**: 1,852 examples
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Multi-level fine-grained constraint following
- **Splits**: train (1,852)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("YuxinJiang/FollowBench")
dataset.save_to_disk("datasets/FollowBench")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/FollowBench")
```

### Notes
- 5 constraint categories: Content, Situation, Style, Format, Example
- Multi-level mechanism incrementally adds constraints
- Features: `example_id`, `category`, `source`, `instruction`, `level`, `target`

---

## Dataset 5: AdvancedIF

### Overview
- **Source**: HuggingFace `facebook/AdvancedIF`
- **Size**: 1,645 prompts
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Complex multi-turn instruction following with rubrics
- **License**: CC-BY-NC-4.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("facebook/AdvancedIF")
dataset.save_to_disk("datasets/AdvancedIF")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/AdvancedIF")
```

### Notes
- Includes inter-conditional instructions
- Features: `conversation_history`, `benchmark_name`, `prompt_metadata`
- Contains expert-curated rubrics for evaluation
