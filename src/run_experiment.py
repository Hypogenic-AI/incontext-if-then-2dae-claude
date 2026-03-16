"""
Run the if-then benchmark against GPT-4.1 via OpenAI API.

Sends each test case to the model and collects responses for verification.
"""

import json
import os
import time
import sys
from datetime import datetime
from openai import OpenAI
from benchmark import generate_all_cases, verify_response, TestCase

# ── Configuration ──
MODEL = "gpt-4.1"
TEMPERATURE = 0
MAX_TOKENS = 512
SEED = 42
RATE_LIMIT_DELAY = 0.2  # seconds between API calls

def run_experiment():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    cases = generate_all_cases()
    print(f"Running {len(cases)} test cases against {MODEL}...")
    print(f"Temperature: {TEMPERATURE}, Max tokens: {MAX_TOKENS}")
    print(f"Start time: {datetime.now().isoformat()}")

    results = []
    responses = []
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

            # Verify
            verification = verify_response(response_text, case)

            result_entry = {
                "case_id": case.case_id,
                "condition_type": case.condition_type,
                "num_rules": case.num_rules,
                "user_message": case.user_message,
                "response": response_text,
                "verification": verification,
                "usage": usage,
            }
            results.append(result_entry)

            # Save incrementally every 50 cases
            if (i + 1) % 50 == 0:
                _save_results(results, errors)

        except Exception as e:
            print(f"  ERROR on case {case.case_id}: {e}")
            errors.append({"case_id": case.case_id, "error": str(e)})
            time.sleep(2)  # Back off on error

        time.sleep(RATE_LIMIT_DELAY)

    # Final save
    _save_results(results, errors)

    print(f"\nCompleted: {len(results)} successful, {len(errors)} errors")
    print(f"End time: {datetime.now().isoformat()}")

    # Print quick summary
    _print_summary(results)

    return results


def _save_results(results, errors):
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    if errors:
        with open("results/experiment_errors.json", "w") as f:
            json.dump(errors, f, indent=2)


def _print_summary(results):
    """Print a quick summary of results."""
    from collections import defaultdict

    by_type = defaultdict(list)
    for r in results:
        by_type[r["condition_type"]].append(r)

    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)

    total_tokens = sum(r["usage"]["total_tokens"] for r in results)
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Estimated cost: ${total_tokens * 0.000005:.2f}")  # rough estimate

    for ctype, type_results in sorted(by_type.items()):
        correct = sum(1 for r in type_results if r["verification"]["all_rules_correct"])
        total = len(type_results)
        print(f"\n{ctype}:")
        print(f"  Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

        # Count FP and FN
        fp_count = 0
        fn_count = 0
        total_rules = 0
        for r in type_results:
            for rule_result in r["verification"]["per_rule_results"]:
                total_rules += 1
                if rule_result["error_type"] == "false_positive":
                    fp_count += 1
                elif rule_result["error_type"] == "false_negative":
                    fn_count += 1
        print(f"  Rule-level: {total_rules} rules, {fp_count} FP, {fn_count} FN")


if __name__ == "__main__":
    run_experiment()
