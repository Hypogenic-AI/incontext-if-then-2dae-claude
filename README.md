# In-Context If-Then Capacity

Measuring LLMs' ability to follow conditional (if-then) instructions provided in system prompts.

## Key Findings

- **If-then accuracy varies significantly by condition type** (p < 0.002): sustained/adversarial rules are hardest (42.9%–78.6%), while if-then-else branching is easiest (90%–100%)
- **False positives outnumber false negatives 2.8:1** (p < 0.001): models over-trigger conditional rules far more than they under-trigger them
- **Model size strongly predicts performance**: GPT-4.1 = 97.2%, GPT-4.1-mini = 87.5%, GPT-4.1-nano = 68.1% case accuracy
- **Even frontier models fail** at adversarial override attempts (78.6%) and cross-language condition leakage
- **Smaller models default to over-compliance**: GPT-4.1-nano includes rule markers even when conditions clearly aren't met

## Benchmark

144 test cases across 7 categories: lexical, semantic, near-miss, sustained, scaling, negation, if-then-else. Verification is fully deterministic (marker string matching).

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy pandas matplotlib seaborn scipy tqdm tabulate

# Run experiments (requires OPENAI_API_KEY)
python src/run_expanded.py gpt-4.1
python src/run_expanded.py gpt-4.1-mini
python src/run_expanded.py gpt-4.1-nano

# Analyze results
python src/analyze.py
```

## File Structure

```
├── REPORT.md                          # Full research report with results
├── planning.md                        # Research plan and methodology
├── src/
│   ├── benchmark.py                   # Initial (easy) benchmark generator
│   ├── benchmark_hard.py              # Hard benchmark generator
│   ├── benchmark_expanded.py          # Full expanded benchmark (7 categories)
│   ├── run_experiment.py              # Runner for initial benchmark
│   ├── run_hard_experiment.py         # Runner for hard benchmark
│   ├── run_expanded.py                # Runner for expanded benchmark
│   └── analyze.py                     # Analysis and visualization
├── results/
│   ├── expanded_gpt-4.1.json          # GPT-4.1 raw results
│   ├── expanded_gpt-4.1-mini.json     # GPT-4.1-mini raw results
│   ├── expanded_gpt-4.1-nano.json     # GPT-4.1-nano raw results
│   ├── analysis_summary.csv           # Summary data
│   └── plots/                         # Generated visualizations
├── literature_review.md               # Literature review
├── resources.md                       # Resource catalog
├── papers/                            # Downloaded papers
├── datasets/                          # Downloaded datasets
└── code/                              # Cloned repositories
```

See [REPORT.md](REPORT.md) for the full research report.
