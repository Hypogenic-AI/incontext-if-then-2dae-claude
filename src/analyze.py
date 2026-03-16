"""
Analysis and visualization of if-then benchmark results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from collections import defaultdict

# Style setup
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
MODEL_LABELS = {"gpt-4.1": "GPT-4.1", "gpt-4.1-mini": "GPT-4.1-mini", "gpt-4.1-nano": "GPT-4.1-nano"}


def load_results():
    """Load all model results into a single list."""
    all_results = []
    for model in MODELS:
        fpath = f"{RESULTS_DIR}/expanded_{model}.json"
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
                all_results.extend(data)
    return all_results


def build_dataframe(results):
    """Convert results to a pandas DataFrame for analysis."""
    rows = []
    for r in results:
        v = r["verification"]
        fp = sum(1 for rr in v["per_rule_results"] if rr["error_type"] == "false_positive")
        fn = sum(1 for rr in v["per_rule_results"] if rr["error_type"] == "false_negative")
        n_rules = len(v["per_rule_results"])
        rules_correct = sum(1 for rr in v["per_rule_results"] if rr["correct"])

        rows.append({
            "model": r["model"],
            "model_label": MODEL_LABELS.get(r["model"], r["model"]),
            "case_id": r["case_id"],
            "category": r["category"],
            "condition_type": r["condition_type"],
            "difficulty": r["difficulty"],
            "num_rules": r["num_rules"],
            "all_correct": v["all_rules_correct"],
            "rule_accuracy": rules_correct / n_rules if n_rules else 1.0,
            "false_positives": fp,
            "false_negatives": fn,
            "total_rules": n_rules,
            "rules_correct": rules_correct,
        })
    return pd.DataFrame(rows)


def plot_category_accuracy(df):
    """Bar chart: accuracy by category for each model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    cat_order = df.groupby("category")["all_correct"].mean().sort_values().index.tolist()
    pivot = df.groupby(["category", "model_label"])["all_correct"].mean().unstack()
    pivot = pivot.reindex(cat_order)

    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Case-Level Accuracy")
    ax.set_xlabel("Condition Category")
    ax.set_title("If-Then Instruction Following Accuracy by Condition Category")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", label_type="edge", fontsize=8,
                     padding=2, labels=[f"{v*100:.0f}%" for v in container.datavalues])

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/category_accuracy.png", bbox_inches="tight")
    plt.close()
    print("Saved: category_accuracy.png")


def plot_fp_fn_breakdown(df):
    """Stacked bar: FP vs FN rates by category for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, model in zip(axes, MODELS):
        model_label = MODEL_LABELS[model]
        mdf = df[df["model"] == model]

        cat_data = mdf.groupby("category").agg(
            fp=("false_positives", "sum"),
            fn=("false_negatives", "sum"),
            total_rules=("total_rules", "sum"),
        )
        cat_data["fp_rate"] = cat_data["fp"] / cat_data["total_rules"]
        cat_data["fn_rate"] = cat_data["fn"] / cat_data["total_rules"]
        cat_data = cat_data.sort_values("fp_rate", ascending=True)

        x = range(len(cat_data))
        ax.barh(list(x), cat_data["fp_rate"], label="False Positive Rate", color="#e74c3c", alpha=0.8)
        ax.barh(list(x), cat_data["fn_rate"], left=cat_data["fp_rate"],
                label="False Negative Rate", color="#3498db", alpha=0.8)
        ax.set_yticks(list(x))
        ax.set_yticklabels(cat_data.index)
        ax.set_xlabel("Error Rate")
        ax.set_title(model_label)
        ax.legend(fontsize=8)

    plt.suptitle("False Positive vs False Negative Rates by Condition Category", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fp_fn_breakdown.png", bbox_inches="tight")
    plt.close()
    print("Saved: fp_fn_breakdown.png")


def plot_scaling(df):
    """Line chart: accuracy vs number of simultaneous rules."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scaling_df = df[df["category"] == "scaling"]
    for model in MODELS:
        mdf = scaling_df[scaling_df["model"] == model]
        grouped = mdf.groupby("num_rules").agg(
            case_acc=("all_correct", "mean"),
            rule_acc=("rule_accuracy", "mean"),
        )
        ax.plot(grouped.index, grouped["rule_acc"], marker="o", linewidth=2,
                label=f"{MODEL_LABELS[model]} (rule-level)")
        ax.plot(grouped.index, grouped["case_acc"], marker="s", linewidth=2,
                linestyle="--", alpha=0.6,
                label=f"{MODEL_LABELS[model]} (case-level)")

    ax.set_xlabel("Number of Simultaneous Rules")
    ax.set_ylabel("Accuracy")
    ax.set_title("If-Then Accuracy vs. Number of Simultaneous Rules")
    ax.set_xticks([4, 8, 12, 16])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scaling.png", bbox_inches="tight")
    plt.close()
    print("Saved: scaling.png")


def plot_difficulty(df):
    """Accuracy by difficulty level."""
    fig, ax = plt.subplots(figsize=(8, 5))

    diff_order = ["easy", "medium", "hard"]
    pivot = df.groupby(["difficulty", "model_label"])["all_correct"].mean().unstack()
    pivot = pivot.reindex(diff_order)

    pivot.plot(kind="bar", ax=ax, width=0.7)
    ax.set_ylabel("Case-Level Accuracy")
    ax.set_xlabel("Difficulty Level")
    ax.set_title("If-Then Accuracy by Difficulty Level")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/difficulty.png", bbox_inches="tight")
    plt.close()
    print("Saved: difficulty.png")


def plot_heatmap(df):
    """Heatmap: model × category accuracy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    pivot = df.groupby(["model_label", "category"])["all_correct"].mean().unstack()
    # Sort columns by overall difficulty
    col_order = pivot.mean().sort_values().index.tolist()
    pivot = pivot[col_order]

    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0.3, vmax=1.0,
                ax=ax, linewidths=0.5, cbar_kws={"label": "Accuracy"})
    ax.set_title("If-Then Instruction Following: Model × Category Accuracy Heatmap")
    ax.set_ylabel("Model")
    ax.set_xlabel("Condition Category")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/heatmap.png", bbox_inches="tight")
    plt.close()
    print("Saved: heatmap.png")


def plot_error_type_by_model(df):
    """Compare overall error types across models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    error_data = []
    for model in MODELS:
        mdf = df[df["model"] == model]
        total_rules = mdf["total_rules"].sum()
        fp = mdf["false_positives"].sum()
        fn = mdf["false_negatives"].sum()
        correct = mdf["rules_correct"].sum()
        error_data.append({
            "Model": MODEL_LABELS[model],
            "Correct": correct / total_rules,
            "False Positive": fp / total_rules,
            "False Negative": fn / total_rules,
        })

    edf = pd.DataFrame(error_data).set_index("Model")
    edf[["Correct", "False Positive", "False Negative"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#2ecc71", "#e74c3c", "#3498db"])
    ax.set_ylabel("Proportion of Rules")
    ax.set_title("Error Type Distribution by Model")
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/error_types.png", bbox_inches="tight")
    plt.close()
    print("Saved: error_types.png")


def run_statistical_tests(df):
    """Run statistical tests on the results."""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # Test 1: Chi-squared test — is accuracy independent of category?
    for model in MODELS:
        mdf = df[df["model"] == model]
        contingency = pd.crosstab(mdf["category"], mdf["all_correct"])
        if contingency.shape[1] == 2:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            print(f"\n[{MODEL_LABELS[model]}] Chi-squared test (accuracy ~ category):")
            print(f"  χ² = {chi2:.2f}, df = {dof}, p = {p:.4f}")
            if p < 0.05:
                print(f"  → SIGNIFICANT: Accuracy varies by category (p < 0.05)")
            else:
                print(f"  → Not significant (p ≥ 0.05)")
        else:
            print(f"\n[{MODEL_LABELS[model]}] Chi-squared test: skipped (need both T/F outcomes)")

    # Test 2: Kruskal-Wallis — accuracy across categories
    for model in MODELS:
        mdf = df[df["model"] == model]
        groups = [g["all_correct"].astype(float).values
                  for _, g in mdf.groupby("category") if len(g) > 1]
        if len(groups) > 1:
            h_stat, p = stats.kruskal(*groups)
            print(f"\n[{MODEL_LABELS[model]}] Kruskal-Wallis (accuracy ~ category):")
            print(f"  H = {h_stat:.2f}, p = {p:.4f}")

    # Test 3: FP vs FN comparison
    print("\n\nFP vs FN comparison (all models combined):")
    total_fp = df["false_positives"].sum()
    total_fn = df["false_negatives"].sum()
    total_errors = total_fp + total_fn
    if total_errors > 0:
        # Binomial test: is FP proportion different from 0.5?
        binom_result = stats.binomtest(total_fp, total_errors, 0.5)
        print(f"  Total FP: {total_fp}, Total FN: {total_fn}")
        print(f"  FP proportion: {total_fp/total_errors:.3f}")
        print(f"  Binomial test p = {binom_result.pvalue:.4f}")
        if binom_result.pvalue < 0.05:
            print(f"  → SIGNIFICANT: FP and FN rates differ")
        else:
            print(f"  → Not significant")

    # Test 4: Model comparison (pairwise McNemar)
    print("\n\nPairwise model comparison (McNemar's test):")
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i+1:]:
            df1 = df[df["model"] == m1].sort_values("case_id")
            df2 = df[df["model"] == m2].sort_values("case_id")
            # Align on case_id
            merged = pd.merge(
                df1[["case_id", "all_correct"]],
                df2[["case_id", "all_correct"]],
                on="case_id", suffixes=("_1", "_2")
            )
            a = merged["all_correct_1"].astype(int).values
            b = merged["all_correct_2"].astype(int).values
            # Discordant pairs
            n01 = ((a == 0) & (b == 1)).sum()  # m1 wrong, m2 right
            n10 = ((a == 1) & (b == 0)).sum()  # m1 right, m2 wrong
            if n01 + n10 > 0:
                mcnemar_stat = (abs(n01 - n10) - 1)**2 / (n01 + n10) if n01 + n10 > 0 else 0
                p_val = stats.chi2.sf(mcnemar_stat, 1)
                print(f"\n  {MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}:")
                print(f"    {MODEL_LABELS[m1]} right, {MODEL_LABELS[m2]} wrong: {n10}")
                print(f"    {MODEL_LABELS[m1]} wrong, {MODEL_LABELS[m2]} right: {n01}")
                print(f"    McNemar χ² = {mcnemar_stat:.2f}, p = {p_val:.4f}")

    return {}


def examine_failures(df, results):
    """Print detailed failure analysis."""
    print("\n" + "="*70)
    print("FAILURE CASE ANALYSIS")
    print("="*70)

    # Get failures for each model
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        failures = [r for r in model_results if not r["verification"]["all_rules_correct"]]
        print(f"\n--- {MODEL_LABELS[model]}: {len(failures)} failures ---")

        by_cat = defaultdict(list)
        for f in failures:
            by_cat[f["category"]].append(f)

        for cat, cat_failures in sorted(by_cat.items()):
            print(f"\n  [{cat}] ({len(cat_failures)} failures):")
            for f in cat_failures[:3]:  # Show first 3
                error_types = [rr["error_type"] for rr in f["verification"]["per_rule_results"]
                               if rr["error_type"]]
                print(f"    Case: {f['case_id']}")
                print(f"    Msg: {f['user_message'][:80]}...")
                print(f"    Errors: {error_types}")
                print(f"    Notes: {f['notes']}")


def generate_summary_table(df):
    """Generate summary tables for the report."""
    print("\n" + "="*70)
    print("SUMMARY TABLE (for REPORT.md)")
    print("="*70)

    # Model × Category accuracy table
    print("\n### Case-Level Accuracy (%) by Model and Category\n")
    pivot = df.groupby(["model_label", "category"])["all_correct"].mean() * 100
    pivot = pivot.unstack().round(1)
    # Add overall column
    overall = df.groupby("model_label")["all_correct"].mean() * 100
    pivot["OVERALL"] = overall.round(1)
    print(pivot.to_markdown())

    # FP/FN table
    print("\n### Error Counts by Model and Category\n")
    for model in MODELS:
        mdf = df[df["model"] == model]
        print(f"\n**{MODEL_LABELS[model]}**")
        cat_errors = mdf.groupby("category").agg(
            N=("case_id", "count"),
            FP=("false_positives", "sum"),
            FN=("false_negatives", "sum"),
        )
        cat_errors["Total_Errors"] = cat_errors["FP"] + cat_errors["FN"]
        print(cat_errors.to_markdown())

    return pivot


if __name__ == "__main__":
    results = load_results()
    df = build_dataframe(results)

    print(f"Total results loaded: {len(df)}")
    print(f"Models: {df['model'].unique()}")
    print(f"Categories: {df['category'].unique()}")

    # Generate all plots
    plot_category_accuracy(df)
    plot_fp_fn_breakdown(df)
    plot_scaling(df)
    plot_difficulty(df)
    plot_heatmap(df)
    plot_error_type_by_model(df)

    # Statistical tests
    run_statistical_tests(df)

    # Failure analysis
    examine_failures(df, results)

    # Summary tables
    summary = generate_summary_table(df)

    # Save analysis data
    df.to_csv(f"{RESULTS_DIR}/analysis_summary.csv", index=False)
    print(f"\nAnalysis complete. Summary saved to {RESULTS_DIR}/analysis_summary.csv")
