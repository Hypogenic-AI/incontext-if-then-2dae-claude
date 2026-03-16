"""
Run the hard if-then benchmark against GPT-4.1.
"""

import json
import os
import time
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from benchmark_hard import generate_all_hard_cases, verify_hard

MODEL = "gpt-4.1"
TEMPERATURE = 0
MAX_TOKENS = 512
SEED = 42
RATE_LIMIT_DELAY = 0.2


def run():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cases = generate_all_hard_cases()
    print(f"Running {len(cases)} hard test cases against {MODEL}...")
    print(f"Start time: {datetime.now().isoformat()}")

    results = []
    errors = []

    for i, case in enumerate(cases):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(cases)} ({100*i/len(cases):.0f}%)")
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": case.system_prompt},
                    {"role": "user", "content": case.user_message},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=SEED,
            )
            response_text = completion.choices[0].message.content or ""
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
            verification = verify_hard(response_text, case)
            results.append({
                "case_id": case.case_id,
                "condition_type": case.condition_type,
                "num_rules": case.num_rules,
                "difficulty": case.difficulty,
                "user_message": case.user_message,
                "response": response_text,
                "verification": verification,
                "usage": usage,
                "notes": case.notes,
            })
            if (i + 1) % 50 == 0:
                _save(results, errors)
        except Exception as e:
            print(f"  ERROR on {case.case_id}: {e}")
            errors.append({"case_id": case.case_id, "error": str(e)})
            time.sleep(2)
        time.sleep(RATE_LIMIT_DELAY)

    _save(results, errors)
    print(f"\nCompleted: {len(results)} OK, {len(errors)} errors")
    print(f"End time: {datetime.now().isoformat()}")
    _summary(results)
    return results


def _save(results, errors):
    with open("results/hard_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    if errors:
        with open("results/hard_experiment_errors.json", "w") as f:
            json.dump(errors, f, indent=2)


def _summary(results):
    by_type = defaultdict(list)
    for r in results:
        by_type[r["condition_type"]].append(r)

    total_tokens = sum(r["usage"]["total_tokens"] for r in results)
    print(f"\n{'='*70}")
    print(f"HARD BENCHMARK SUMMARY  |  Total tokens: {total_tokens:,}")
    print(f"{'='*70}")
    print(f"{'Type':<20} {'Correct':>8} {'Total':>6} {'Acc%':>6} {'FP':>4} {'FN':>4}")
    print("-" * 70)

    overall_correct = 0
    overall_total = 0

    for ctype in sorted(by_type.keys()):
        type_results = by_type[ctype]
        correct = sum(1 for r in type_results if r["verification"]["all_rules_correct"])
        total = len(type_results)
        overall_correct += correct
        overall_total += total

        fp = sum(1 for r in type_results
                 for rr in r["verification"]["per_rule_results"]
                 if rr["error_type"] == "false_positive")
        fn = sum(1 for r in type_results
                 for rr in r["verification"]["per_rule_results"]
                 if rr["error_type"] == "false_negative")
        print(f"{ctype:<20} {correct:>8} {total:>6} {100*correct/total:>5.1f}% {fp:>4} {fn:>4}")

    print("-" * 70)
    print(f"{'OVERALL':<20} {overall_correct:>8} {overall_total:>6} "
          f"{100*overall_correct/overall_total:>5.1f}%")


if __name__ == "__main__":
    run()
