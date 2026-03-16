# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "In-context If-Then Capacity" — measuring how consistently LLMs follow conditional (if-then) instructions across different instruction types.

## Papers
Total papers downloaded: 18

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| IFEval | Zhou et al. (Google) | 2023 | papers/2311.07911_ifeval_instruction_following_eval.pdf | Foundational IF benchmark, no conditional logic |
| ComplexBench | Wen et al. (Tsinghua) | 2024 | papers/2407.03978_complexbench_multiple_constraints.pdf | **Selection = if-then-else branching** |
| TOD-ProcBench | Ghazarian et al. (Amazon) | 2025 | papers/2511.15976_tod_procbench_condition_action_instructions.pdf | **Condition-action if-then structures** |
| SIFo | Chen et al. (UvA) | 2024 | papers/2406.19999_sifo_sequential_instruction_following.pdf | Sequential deps, implicit conditionality |
| RuleBERT | Saeed et al. | 2021 | papers/2109.13006_rulebert_soft_rules_plm.pdf | **Soft Horn rule (if-then) following** |
| When Models Can't Follow | Young et al. | 2025 | papers/2510.18892_when_models_cant_follow_256_llms.pdf | 256 LLMs tested |
| AdvancedIF | He et al. (Meta) | 2025 | papers/2511.10507_advancedif_rubric_benchmark.pdf | Inter-conditional instructions |
| Multi-IF | He et al. | 2024 | papers/2410.15553_multi_if_multi_turn.pdf | Multi-turn degradation |
| ICL Sufficient? | Zhao et al. (EPFL) | 2024 | papers/2405.19874_icl_sufficient_instruction_following.pdf | ICL vs IFT for instruction following |
| Incomplete Loop | Liu et al. (CMU/MIT) | 2024 | papers/2404.03028_incomplete_loop_icl_instruction.pdf | Format affects instruction adherence |
| NSVIF | Su et al. (Microsoft) | 2026 | papers/2601.17789_nsvif_neuro_symbolic_verification.pdf | Constraint satisfaction verification |
| IF-CRITIC | Wen et al. | 2025 | papers/2511.01014_if_critic_fine_grained.pdf | Checklist-based evaluation |
| LLMBar | Zeng et al. | 2023 | papers/2310.07641_llmbar_evaluating_instruction_following.pdf | Meta-evaluation of IF evaluators |
| Instruction Gap | Tripathi et al. | 2025 | papers/2601.03269_instruction_gap_enterprise.pdf | Enterprise instruction adherence |
| IF Survey | Lou et al. | 2023 | papers/2303.10475_survey_instruction_following.pdf | Comprehensive IF survey |
| DMN-Guided Prompting | Abedi & Jalali | 2025 | papers/2505.11701_dmn_guided_prompting_decision_logic.pdf | Decision logic for LLM control |
| FollowBench | Various | 2024 | papers/2410.12163_followbench_multi_level_constraint.pdf | Multi-level constraints |
| Can LLMs Follow Rules | Various | 2023 | papers/2308.01862_llm_rules_following.pdf | Rule following & prompt injection |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 5

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| IFEval | google/IFEval | 541 prompts | Verifiable constraints | datasets/IFEval/ | Baseline (unconditional) |
| ComplexBench | thu-coai/ComplexBench | 1,150 instructions | Multi-constraint composition | datasets/ComplexBench_data_final.json | **Has Selection (if-then)** |
| SIFo | shin-ee-chen/SIFo | 800 samples | Sequential instructions | datasets/SIFo/ | Security task has implicit conditionals |
| FollowBench | YuxinJiang/FollowBench | 1,852 examples | Multi-level constraints | datasets/FollowBench/ | Cumulative constraint levels |
| AdvancedIF | facebook/AdvancedIF | 1,645 prompts | Complex multi-turn | datasets/AdvancedIF/ | Inter-conditional instructions |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| ComplexBench | github.com/thu-coai/ComplexBench | IF benchmark with Selection composition | code/ComplexBench/ | Evaluation scripts included |
| SIFo | github.com/shin-ee-chen/SIFo | Sequential IF benchmark | code/SIFo/ | 4 task evaluation scripts |
| RuleBert | github.com/MhmdSaiid/RuleBert | Soft Horn rule fine-tuning | code/RuleBert/ | Full data on Zenodo (25GB) |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used arXiv API with 8 search queries covering: conditional instruction following, if-then rules, constraint following, system prompt adherence, in-context learning, and instruction following benchmarks
- Searched HuggingFace Hub for related datasets
- Reviewed 32 papers from search results, selected 18 most relevant for download
- Deep-read 6 key papers using PDF chunker (all pages)

### Selection Criteria
- **Primary**: Papers that explicitly test conditional/if-then instruction following (ComplexBench, TOD-ProcBench, RuleBERT)
- **Secondary**: Instruction following benchmarks and evaluation methods that provide context and baselines (IFEval, SIFo, AdvancedIF)
- **Tertiary**: ICL studies and surveys that inform methodology (ICL Sufficient, Incomplete Loop, IF Survey)

### Key Finding
No existing benchmark specifically and systematically measures the in-context if-then capacity of LLMs. ComplexBench's "Selection" type comes closest but is part of a broader benchmark. TOD-ProcBench tests condition-action pairs but is domain-specific (customer service). This gap validates the research hypothesis.

### Gaps and Workarounds
- TOD-ProcBench dataset requires accessing Amazon Science portal (not directly downloadable via API)
- RuleBERT's full training data (25GB) was not downloaded; only code repo cloned
- No single dataset exists that systematically varies if-then instruction types, formats, and complexity — this will need to be created for the experiment

## Recommendations for Experiment Design

### 1. Primary Dataset Strategy
Create a **custom if-then benchmark** that systematically varies:
- **Condition type**: content-based ("if the text mentions X"), input-based ("if the user asks about Y"), format-based ("if your response exceeds N words")
- **Number of simultaneous rules**: 1, 2, 4, 8 if-then rules
- **Nesting depth**: flat (if-then), nested (if-then within if-then), chained
- **Rule format**: natural language, structured bullets, JSON, pseudocode
- **Include distractors**: irrelevant conditions that should not trigger

Use IFEval's deterministic verification approach where possible.

### 2. Baseline Comparison
- Compare if-then instruction following against unconditional instruction following using IFEval prompts as control
- Use ComplexBench's Selection subset as a reference benchmark

### 3. Baseline Methods
- Test across model families: GPT-4o, Claude 3.5 Sonnet, Llama 3 70B/8B, Qwen2 72B/7B
- Compare zero-shot vs few-shot (with if-then rule examples)
- Test structured prompting (JSON format rules) vs natural language rules

### 4. Evaluation Metrics
- **Branch identification accuracy**: Did the model correctly determine which condition applies?
- **Action execution accuracy**: Given correct branch, was the action correct?
- **Overall if-then accuracy**: End-to-end correctness
- **Consistency across runs**: Same condition → same response
- **Degradation curve**: How accuracy changes with number of rules

### 5. Code to Adapt/Reuse
- **ComplexBench** evaluation scripts for Selection-type assessment
- **IFEval** deterministic verification functions for baseline constraints
- **SIFo** evaluation methodology for sequential/dependent instructions
