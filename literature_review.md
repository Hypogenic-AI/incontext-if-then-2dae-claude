# Literature Review: In-context If-Then Capacity

## Research Area Overview

This review examines the ability of large language models (LLMs) to follow conditional (if-then) instructions provided in context. While instruction following has been extensively studied, the specific capacity of LLMs to handle conditional branching logic — "if condition X holds, then do Y; otherwise do Z" — remains underexplored. Most existing benchmarks test unconditional constraints (format, length, keywords) rather than conditional rules. This gap is significant because real-world applications (customer service workflows, decision-making, policy enforcement) heavily rely on conditional instructions.

---

## Key Papers

### Paper 1: IFEval — Instruction-Following Eval for Large Language Models
- **Authors**: Jeffrey Zhou, Tianjian Lu, Swaroop Mishra et al. (Google)
- **Year**: 2023
- **Source**: arXiv:2311.07911
- **Key Contribution**: Foundational benchmark with 25 verifiable instruction types and 541 prompts. All verification is deterministic (no LLM judge needed).
- **Methodology**: Each prompt contains 1-3 verifiable instructions (keyword inclusion, format constraints, length requirements, etc.). Reports strict and loose accuracy at both prompt and instruction level.
- **Datasets Used**: Custom 541-prompt dataset
- **Results**: GPT-4 achieves 76.89% prompt-level strict accuracy; PaLM 2 Small achieves 43.07%.
- **Code Available**: Yes — https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **Relevance**: IFEval does NOT test conditional if-then instructions — all 25 instruction types are unconditional constraints. This represents a clear gap that our research addresses.

### Paper 2: ComplexBench — Benchmarking Complex Instruction-Following with Multiple Constraints Composition
- **Authors**: Bosi Wen, Pei Ke et al. (Tsinghua, NeurIPS 2024)
- **Year**: 2024
- **Source**: arXiv:2407.03978
- **Key Contribution**: Introduces a hierarchical taxonomy of constraint compositions including **Selection** — explicit if-then-else branching logic. This is the most directly relevant benchmark for our research.
- **Methodology**: 1,150 instructions with 4 constraint types (Lexical, Format, Semantic, Utility), 19 dimensions, and 4 composition types (Single, And, Chain, Selection). Uses Rule-Augmented LLM evaluation with dependency graphs.
- **Datasets Used**: Custom manually constructed dataset (Chinese + English)
- **Results**: GPT-4 achieves 0.800 overall but only 0.765 on Selection and 0.675 on Selection+Chain — showing conditional branching is significantly harder than flat constraints (0.881 for And).
- **Code Available**: Yes — https://github.com/thu-coai/ComplexBench
- **Relevance**: **HIGHLY RELEVANT.** The Selection composition type directly tests if-then conditional branching. Key finding: models struggle more with conditional selection than with parallel constraints.

### Paper 3: TOD-ProcBench — Benchmarking Complex Instruction-Following in Task-Oriented Dialogues
- **Authors**: Sarik Ghazarian et al. (Amazon, NeurIPS 2025 Workshop)
- **Year**: 2025
- **Source**: arXiv:2511.15976
- **Key Contribution**: Operationalizes if-then instructions as multi-level condition-action pairs in customer service dialogues. Tests nested conditional structures with up to 4 levels of depth.
- **Methodology**: Three instruction formats: nested if-then (f1), flattened if-then (f2), JSON mappings (f3). Three tasks: instruction retrieval + action prediction, compliance evaluation, compliant response generation. Uses 55 user intents from ABCD dataset.
- **Datasets Used**: Derived from ABCD dataset; 1,004 test conversations, 769 conversation-instruction pairs.
- **Results**: Best model (Claude 3.7 Sonnet) achieves only 39.3% on combined instruction retrieval + action prediction. Compliance evaluation reaches ~76%. All models struggle significantly with multi-level conditional logic.
- **Code Available**: Data at Amazon Science portal
- **Relevance**: **HIGHLY RELEVANT.** Directly tests condition-action if-then structures in realistic dialogue scenarios. Demonstrates that even frontier models fail at multi-level conditional reasoning.

### Paper 4: SIFo — Sequential Instruction Following Benchmark
- **Authors**: Xinyi Chen, Baohao Liao et al. (University of Amsterdam)
- **Year**: 2024
- **Source**: arXiv:2406.19999
- **Key Contribution**: Tests causally dependent sequential instructions across 4 domains (text modification, QA, math, security rules). Security Rules task involves implicit conditional reasoning.
- **Methodology**: 800 samples, 200 per task. Instructions are causally chained — only the final output needs checking. Verifiable without LLM judges.
- **Datasets Used**: Derived from SQuAD and GSM8K; 800 total samples
- **Results**: Even GPT-4 and Claude-3 Opus achieve only 42.5% and 34.0% on text modification. All models show monotonic accuracy decline as sequence length increases.
- **Code Available**: Yes — https://github.com/shin-ee-chen/SIFo
- **Relevance**: The Security Rules task tests implicit conditionality (if password is X, then command is valid). Sequential dependency creates implicit if-then reasoning chains.

### Paper 5: RuleBERT — Teaching Soft Rules to Pre-trained Language Models
- **Authors**: Mohammed Saeed, Naser Ahmadi et al. (EMNLP 2021)
- **Year**: 2021
- **Source**: arXiv:2109.13006
- **Key Contribution**: Fine-tunes PLMs on soft Horn rules expressed in natural language. Directly tests if-then deductive reasoning with probabilistic rules.
- **Methodology**: Synthetic dataset from 161 Horn rules mined from DBpedia. Classification task: given facts + rules + hypothesis, predict probability. Tests single rules, conflicting rules, and chained rules (up to depth 5).
- **Datasets Used**: Synthetic dataset generated with LPMLN reasoner (25GB full dataset on Zenodo)
- **Results**: Fine-tuned RoBERTa-LARGE achieves high accuracy even on unseen rules. Generalizes well to novel if-then patterns. Transfers logical notions to external datasets.
- **Code Available**: Yes — https://github.com/MhmdSaiid/RuleBert
- **Relevance**: Directly tests if-then rule following, but through fine-tuning rather than in-context learning. Provides methodology for evaluating probabilistic rule adherence.

### Paper 6: When Models Can't Follow — Testing Instruction Adherence Across 256 LLMs
- **Authors**: Richard J. Young et al.
- **Year**: 2025
- **Source**: arXiv:2510.18892
- **Key Contribution**: Large-scale evaluation of 256 LLMs on 20 instruction-following prompts. Identifies consistent failure modes across models.
- **Results**: Reveals specific instruction types that are consistently challenging across model families.
- **Relevance**: Provides broad empirical evidence of instruction-following failures at scale, though does not specifically isolate conditional if-then instructions.

### Paper 7: Is In-Context Learning Sufficient for Instruction Following in LLMs?
- **Authors**: Hao Zhao et al. (EPFL, ICLR 2025)
- **Year**: 2024
- **Source**: arXiv:2405.19874
- **Key Contribution**: Shows that ICL alignment (URIAL) consistently underperforms instruction fine-tuning on MT-Bench. Decoding parameters are a major overlooked factor.
- **Results**: ICL plateaus after 10-30 examples. ICL and IFT are roughly equivalent for single-turn but IFT generalizes better for multi-turn.
- **Code Available**: Yes — https://github.com/tml-epfl/icl-alignment
- **Relevance**: Important background on ICL capabilities for instruction following, but does not test conditional logic specifically.

### Paper 8: An Incomplete Loop — Instruction Inference, Instruction Following, and In-context Learning
- **Authors**: Emmy Liu, Graham Neubig, Jacob Andreas (CMU/MIT, 2024)
- **Year**: 2024
- **Source**: arXiv:2404.03028
- **Key Contribution**: Reveals dissociation between instruction following, few-shot learning, and instruction inference. Models can succeed with examples while failing with descriptions.
- **Relevance**: Suggests that the format in which if-then rules are presented (examples vs. explicit rules) may significantly affect adherence.

### Paper 9: AdvancedIF — Rubric-Based Benchmarking for LLM Instruction Following
- **Authors**: Yun He et al. (Meta, 2025)
- **Year**: 2025
- **Source**: arXiv:2511.10507
- **Key Contribution**: 1,600+ prompts with expert-curated rubrics assessing complex, multi-turn, and system-level instructions. Includes inter-conditional instructions.
- **Code Available**: Dataset on HuggingFace (facebook/AdvancedIF)
- **Relevance**: Includes conditional instruction types; provides rubric-based evaluation framework.

### Paper 10: LLM Instruction Following Survey
- **Authors**: Renze Lou, Kai Zhang, Wenpeng Yin
- **Year**: 2023
- **Source**: arXiv:2303.10475
- **Key Contribution**: First comprehensive survey on instruction following. Covers instruction types, modeling approaches, datasets, evaluation metrics, and challenges.
- **Relevance**: Provides taxonomic framework for understanding where conditional if-then instructions fit in the broader instruction-following landscape.

### Paper 11: NSVIF — Neuro-Symbolic Verification of Instruction Following
- **Authors**: Yiming Su et al. (Microsoft, 2026)
- **Year**: 2026
- **Source**: arXiv:2601.17789
- **Key Contribution**: Formulates instruction-following verification as constraint satisfaction. Models instructions as both logical and semantic constraints with a unified solver.
- **Relevance**: Provides verification framework that could be adapted for evaluating if-then instruction adherence.

### Paper 12: IF-CRITIC — Fine-Grained LLM Critic for Instruction-Following Evaluation
- **Authors**: Bosi Wen et al. (Tsinghua, 2025)
- **Year**: 2025
- **Source**: arXiv:2511.01014
- **Key Contribution**: Decomposes instructions into constraint checklists for fine-grained evaluation. Achieves better performance than o4-mini and Gemini-3-Pro as judges.
- **Relevance**: Provides evaluation methodology applicable to decomposing and verifying conditional instructions.

---

## Common Methodologies

### Evaluation Approaches
- **Deterministic verification**: Used in IFEval, SIFo — rule-based checking of output against constraints. Most reliable but limited to verifiable instructions.
- **LLM-as-Judge**: Used in ComplexBench (RAL), AdvancedIF (RIFL), IF-CRITIC. More flexible but introduces evaluator variance.
- **Rule-Augmented LLM evaluation**: ComplexBench's approach — decompose into yes/no questions, apply rules where possible, use LLM judge only for semantic questions.
- **Constraint satisfaction**: NSVIF's approach — formalize instructions as logical+semantic constraints, solve with unified solver.

### Instruction Composition Taxonomies
- **IFEval**: Flat, unconditional constraints (25 types)
- **ComplexBench**: And, Chain, Selection (if-then-else), nested compositions
- **TOD-ProcBench**: Single, And, Or, Chain, Selection, Nesting (6 types)
- **SIFo**: Causally dependent sequential chains

---

## Standard Baselines

Models commonly evaluated across papers:
- **GPT-4 / GPT-4o**: Consistently top performer across benchmarks
- **Claude 3/3.5/3.7 Sonnet/Opus**: Strong performance, especially on compliance tasks
- **Llama 3 (8B, 70B)**: Representative open-source baseline
- **Qwen2 (7B, 72B)**: Strong open-source alternative
- **Mistral (7B)**: Lightweight baseline

---

## Evaluation Metrics

- **Prompt-level accuracy**: % of prompts where ALL constraints are satisfied (strict)
- **Instruction-level accuracy**: % of individual instructions followed
- **DRFR (Dependency-aware Rule Following Rate)**: ComplexBench's metric accounting for composition structure
- **Compliance rate**: Binary correct/incorrect for condition-action adherence (TOD-ProcBench)
- **Sample-level accuracy (Acc_S)**: SIFo's metric checking final output correctness

---

## Datasets in the Literature

| Dataset | Source | Size | Task | If-Then? |
|---------|--------|------|------|----------|
| IFEval | Google | 541 prompts | Verifiable constraints | No |
| ComplexBench | Tsinghua | 1,150 instructions | Multi-constraint composition | **Yes (Selection)** |
| TOD-ProcBench | Amazon | 1,004 conversations | Condition-action dialogues | **Yes (core focus)** |
| SIFo | UvA | 800 samples | Sequential instructions | Implicit |
| FollowBench | - | 1,852 examples | Multi-level constraints | Partial |
| AdvancedIF | Meta | 1,645 prompts | Complex multi-turn | Yes (inter-conditional) |
| RuleBERT data | Zenodo | 25GB | Soft Horn rule reasoning | **Yes (core focus)** |

---

## Gaps and Opportunities

1. **No dedicated in-context if-then benchmark**: While ComplexBench includes Selection and TOD-ProcBench tests condition-action pairs, no benchmark systematically isolates and measures the capacity to follow conditional if-then rules provided purely in-context (system prompt or user message).

2. **Limited taxonomy of conditional types**: Existing work conflates different types of conditions (content-dependent, input-dependent, state-dependent, negated conditions, nested conditions). A systematic taxonomy would enable finer-grained measurement.

3. **Format effects on conditional adherence**: The "Incomplete Loop" paper shows models respond differently to rules given as examples vs. descriptions. How if-then rules are formatted (natural language, structured, JSON, pseudocode) likely affects adherence, but this hasn't been systematically studied.

4. **Inconsistency across condition types**: The research hypothesis suggests adherence varies by instruction type. Existing evidence (ComplexBench: Selection < And; TOD-ProcBench: multi-level conditions harder) supports this but hasn't been systematically categorized.

5. **Scale of conditions**: How does performance degrade with number of simultaneous if-then rules? TOD-ProcBench shows 4-level nesting is challenging, but systematic scaling studies are lacking.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **IFEval** (google/IFEval) — as a baseline for unconditional instruction following
2. **ComplexBench** — for its Selection composition type (direct if-then testing)
3. **Custom if-then benchmark** — create a targeted dataset that systematically varies:
   - Condition type (content-based, input-based, format-based)
   - Number of conditions (1, 2, 4, 8)
   - Nesting depth (1, 2, 3 levels)
   - Rule format (natural language, structured, JSON)

### Recommended Baselines
- GPT-4o, Claude 3.5 Sonnet, Llama 3 70B, Qwen2 72B (closed + open source)
- Compare against unconditional instruction following (IFEval-style) as control

### Recommended Metrics
- **Condition identification accuracy**: Did the model correctly identify which branch applies?
- **Action execution accuracy**: Given correct branch identification, was the action correct?
- **Overall if-then accuracy**: End-to-end correctness
- **Consistency**: Same condition should yield same response across runs
- Use deterministic verification where possible (IFEval-style)

### Methodological Considerations
- Separate condition evaluation from action execution to identify where failures occur
- Test both "if X then Y" and "if X then Y else Z" structures
- Include distractor conditions to test selective attention
- Vary the position of if-then rules in the prompt (beginning, middle, end)
- Test whether models follow the spirit vs. letter of conditional rules
