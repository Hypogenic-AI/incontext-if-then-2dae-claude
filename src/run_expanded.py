"""
Run the expanded benchmark against one or more models.
"""

import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from benchmark_expanded import generate_all, verify

TEMPERATURE = 0
MAX_TOKENS = 512
SEED = 42
RATE_LIMIT_DELAY = 0.15


def run_model(model_name: str, output_file: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cases = generate_all()
    print(f"Running {len(cases)} cases against {model_name}...")
    print(f"Start: {datetime.now().isoformat()}")

    results = []
    errors = []

    for i, case in enumerate(cases):
        if i % 25 == 0:
            print(f"  [{model_name}] {i}/{len(cases)} ({100*i/len(cases):.0f}%)")
        try:
            completion = client.chat.completions.create(
                model=model_name,
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
            v = verify(response_text, case)
            results.append({
                "case_id": case.case_id,
                "category": case.category,
                "condition_type": case.condition_type,
                "num_rules": case.num_rules,
                "difficulty": case.difficulty,
                "user_message": case.user_message,
                "system_prompt_len": len(case.system_prompt),
                "response": response_text,
                "verification": v,
                "usage": usage,
                "notes": case.notes,
                "model": model_name,
            })
        except Exception as e:
            print(f"  ERROR on {case.case_id}: {e}")
            errors.append({"case_id": case.case_id, "error": str(e)})
            time.sleep(3)
        time.sleep(RATE_LIMIT_DELAY)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{model_name}] Done: {len(results)} OK, {len(errors)} errors")
    print(f"End: {datetime.now().isoformat()}")
    print_summary(results, model_name)
    return results


def print_summary(results, model_name):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    total_tokens = sum(r["usage"]["total_tokens"] for r in results)
    print(f"\n{'='*75}")
    print(f"{model_name} RESULTS  |  Tokens: {total_tokens:,}")
    print(f"{'='*75}")
    print(f"{'Category':<20} {'Case Acc':>10} {'Rule Acc':>10} {'FP':>5} {'FN':>5} {'N':>5}")
    print("-" * 75)

    overall_correct = 0
    overall_total = 0
    overall_rules_correct = 0
    overall_rules_total = 0

    for cat in sorted(by_cat.keys()):
        cat_results = by_cat[cat]
        correct = sum(1 for r in cat_results if r["verification"]["all_rules_correct"])
        total = len(cat_results)
        overall_correct += correct
        overall_total += total

        fp = fn = rules_ok = rules_total = 0
        for r in cat_results:
            for rr in r["verification"]["per_rule_results"]:
                rules_total += 1
                if rr["correct"]:
                    rules_ok += 1
                if rr["error_type"] == "false_positive":
                    fp += 1
                elif rr["error_type"] == "false_negative":
                    fn += 1
        overall_rules_correct += rules_ok
        overall_rules_total += rules_total

        rule_acc = 100 * rules_ok / rules_total if rules_total else 100
        case_acc = 100 * correct / total if total else 100
        print(f"{cat:<20} {case_acc:>9.1f}% {rule_acc:>9.1f}% {fp:>5} {fn:>5} {total:>5}")

    print("-" * 75)
    oa = 100 * overall_correct / overall_total if overall_total else 100
    ora = 100 * overall_rules_correct / overall_rules_total if overall_rules_total else 100
    print(f"{'OVERALL':<20} {oa:>9.1f}% {ora:>9.1f}%")
    print()


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4.1"
    output = f"results/expanded_{model.replace('/', '_')}.json"
    run_model(model, output)
