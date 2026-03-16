# Research Plan: In-context If-Then Capacity

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used in agentic and workflow settings where they must follow conditional rules: "if the user mentions X, respond with Y", "if the input contains personal data, redact it". Failure to follow these conditional instructions leads to real-world failures in customer service, content moderation, and safety-critical applications. Understanding *where* and *why* models fail at conditional instruction following is essential for building reliable LLM systems.

### Gap in Existing Work
Based on the literature review:
- **IFEval** tests only unconditional constraints (format, length, keywords) — no if-then logic.
- **ComplexBench** includes a "Selection" composition type (if-then-else) but bundles it with broader evaluation; it found Selection accuracy (0.765) is lower than And (0.881), but didn't systematically vary condition types.
- **TOD-ProcBench** tests condition-action pairs but only in customer service dialogues.
- **No existing benchmark** systematically isolates and measures if-then capacity across different condition types, rule counts, and formats.

### Our Novel Contribution
We create a **controlled benchmark** that systematically varies:
1. **Condition type** (content-trigger, format-trigger, counting-trigger, negation-trigger)
2. **Number of simultaneous rules** (1, 2, 4)
3. **Action type** (lexical insertion, format change, content modification)

We measure both **false positives** (rule fires when condition is absent) and **false negatives** (rule doesn't fire when condition is present), directly addressing the user's observation about misfires. We test on a real frontier model (GPT-4.1) with deterministic verification.

### Experiment Justification
- **Experiment 1 (Condition Type Variation)**: Tests whether models follow different types of conditions with different reliability. This directly tests the hypothesis that if-then capacity is inconsistent across instruction types.
- **Experiment 2 (Rule Count Scaling)**: Tests how performance degrades as more simultaneous if-then rules are active. Measures capacity limits.
- **Experiment 3 (False Positive vs False Negative Analysis)**: Separates the two failure modes to understand whether models tend to over-trigger or under-trigger conditional rules.

## Research Question
Can we measure LLMs' capacity to follow conditional (if-then) instructions in context, and does this capacity vary systematically across different types of conditions?

## Hypothesis Decomposition
1. **H1**: If-then instruction following accuracy varies significantly across condition types (content-based vs. format-based vs. counting-based vs. negation-based).
2. **H2**: Accuracy degrades as the number of simultaneous if-then rules increases.
3. **H3**: False positive and false negative rates differ across condition types, indicating distinct failure modes.

## Proposed Methodology

### Approach
Create a programmatic benchmark where each test case consists of:
- A **system prompt** containing 1-4 if-then rules
- A **user message** that either triggers or does not trigger each rule
- **Deterministic verification** of whether the model's output correctly followed/didn't follow each rule

This design enables precise measurement of true positives, true negatives, false positives, and false negatives for each rule type.

### Condition Types (Independent Variable)
1. **Content-trigger**: "If the user mentions [topic], include [phrase] in your response"
2. **Format-trigger**: "If the user asks a question, respond in bullet points"
3. **Counting-trigger**: "If the user's message has more than 20 words, add a summary at the end"
4. **Negation-trigger**: "If the user does NOT mention [topic], do NOT include [phrase]"

### Action Types (Verifiable Actions)
- Include/exclude a specific keyword or phrase
- Use/don't use a specific format (bullets, numbered list)
- Add/omit a specific section (summary, disclaimer)

### Experimental Steps
1. Generate 40 test cases per condition type × trigger/no-trigger = 320 cases for single-rule
2. Generate multi-rule cases (2 rules: 80 cases, 4 rules: 80 cases) = 160 cases
3. Total: ~480 test cases
4. Run each through GPT-4.1 API with temperature=0
5. Verify outputs deterministically (regex/string matching)
6. Compute accuracy, FPR, FNR per condition type and rule count

### Baselines
- **Unconditional baseline**: Same actions without conditions ("Always include [phrase]") — measures action execution ability independent of condition evaluation
- **Random baseline**: Expected accuracy if model ignores conditions entirely

### Evaluation Metrics
- **Condition-following accuracy**: % of cases where the model correctly triggers or doesn't trigger the rule
- **False positive rate (FPR)**: % of no-trigger cases where the rule fires anyway
- **False negative rate (FNR)**: % of trigger cases where the rule doesn't fire
- **Multi-rule accuracy**: % of cases where ALL rules are correctly handled

### Statistical Analysis Plan
- Chi-squared test for independence between condition type and accuracy
- Cochran's Q test for comparing accuracy across condition types (matched samples)
- McNemar's test for pairwise comparisons between condition types
- 95% confidence intervals via Wilson score interval
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- Content-trigger conditions should be easiest (most similar to training distribution)
- Negation-trigger conditions should be hardest (negation is known to be difficult for LLMs)
- Accuracy should decrease with more simultaneous rules
- FPR and FNR should differ across condition types

## Timeline and Milestones
1. Planning: 20 min ✓
2. Benchmark generation: 30 min
3. API experiments: 45 min
4. Analysis & visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
- API rate limits → use batching with delays
- Verification edge cases → use strict regex patterns with fallback manual inspection
- Cost → ~480 calls × ~500 tokens avg = ~240K tokens ≈ $2-5

## Success Criteria
- Clear evidence of variance across condition types (or clear evidence of no variance)
- Statistical significance of any differences found
- Actionable insights about which condition types are most/least reliable
