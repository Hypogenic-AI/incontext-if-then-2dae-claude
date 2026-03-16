# Cloned Repositories

## Repo 1: ComplexBench
- **URL**: https://github.com/thu-coai/ComplexBench
- **Purpose**: Benchmark for complex instruction following with constraint composition (And, Chain, Selection). The Selection type directly tests if-then-else conditional branching.
- **Location**: code/ComplexBench/
- **Key files**:
  - `data/data_final.json` — Main dataset (1,150 instructions)
  - `evaluate.py` — Evaluation script using Rule-Augmented LLM judging
  - `score.py` — Scoring with dependency-aware aggregation
- **Dependencies**: OpenAI API (for LLM judge evaluation)
- **Notes**: NeurIPS 2024. The "Selection" composition type is the most directly relevant component — it tests whether models can correctly identify which branch of an if-then-else to execute.

## Repo 2: SIFo
- **URL**: https://github.com/shin-ee-chen/SIFo
- **Purpose**: Sequential Instruction Following benchmark. Tests causally dependent instruction chains across 4 domains.
- **Location**: code/SIFo/
- **Key files**:
  - `data/` — 4 JSONL files (math, qa, security, text_modification)
  - `evaluation/` — Evaluation scripts for each task
  - `run_*.py` — Model inference scripts
- **Dependencies**: transformers, vllm (for model inference)
- **Notes**: The Security Rules task involves implicit conditional reasoning. 800 samples total, verifiable without LLM judges.

## Repo 3: RuleBert
- **URL**: https://github.com/MhmdSaiid/RuleBert
- **Purpose**: Fine-tuning PLMs on soft Horn rules (if-then logical rules) for deductive reasoning.
- **Location**: code/RuleBert/
- **Key files**:
  - `RuleBert.py` — Main fine-tuning script
  - `rules/` — Horn rule definitions
  - `data/` — Sample data (full 25GB dataset on Zenodo)
- **Dependencies**: transformers, pytorch
- **Notes**: EMNLP 2021. Tests if-then rule following through fine-tuning (not in-context). Full dataset at zenodo.org/record/5644677. Useful for understanding rule following methodology and evaluation.
