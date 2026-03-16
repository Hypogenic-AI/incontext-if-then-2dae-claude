"""
Benchmark generator for measuring in-context if-then instruction following capacity.

Creates test cases that systematically vary:
- Condition type (content, format, counting, negation)
- Whether the condition should trigger (True/False)
- Number of simultaneous rules (1, 2, 4)
"""

import json
import random
import itertools
from dataclasses import dataclass, asdict
from typing import Callable

random.seed(42)

# ── Topics and phrases for condition construction ──

TOPICS = [
    ("dogs", "dogs"),
    ("weather", "weather"),
    ("cooking", "cooking"),
    ("travel", "travel"),
    ("music", "music"),
    ("sports", "sports"),
    ("technology", "technology"),
    ("gardening", "gardening"),
    ("history", "history"),
    ("movies", "movies"),
]

MARKER_PHRASES = [
    "🔵BLUE_MARKER🔵",
    "🔴RED_MARKER🔴",
    "🟢GREEN_MARKER🟢",
    "🟡YELLOW_MARKER🟡",
]

# Using distinctive markers makes verification deterministic and unambiguous.

TRIGGER_MESSAGES = {
    "dogs": [
        "Tell me about the best breeds of dogs for families.",
        "I love playing fetch with my dogs in the park.",
        "What are some tips for training dogs?",
    ],
    "weather": [
        "What will the weather be like this weekend?",
        "I heard the weather is going to be stormy tomorrow.",
        "Can you explain how weather patterns form?",
    ],
    "cooking": [
        "What's a good recipe for cooking pasta?",
        "I've been cooking a lot more at home lately.",
        "Tell me about cooking techniques for beginners.",
    ],
    "travel": [
        "I'm planning some travel to Japan next month.",
        "What are the best travel destinations in Europe?",
        "Can you give me travel tips for budget backpacking?",
    ],
    "music": [
        "I've been listening to a lot of jazz music recently.",
        "What are the best music festivals this summer?",
        "Can you recommend some classical music for studying?",
    ],
    "sports": [
        "What sports are most popular worldwide?",
        "I just started playing sports after years of inactivity.",
        "Tell me about the history of Olympic sports.",
    ],
    "technology": [
        "What new technology trends should I watch for?",
        "How has technology changed education in the last decade?",
        "Tell me about the latest technology in renewable energy.",
    ],
    "gardening": [
        "I need gardening tips for growing tomatoes.",
        "What tools do I need to start gardening?",
        "Tell me about urban gardening in small spaces.",
    ],
    "history": [
        "What are the most important events in ancient history?",
        "I've been reading about the history of the Roman Empire.",
        "Tell me about the history of mathematics.",
    ],
    "movies": [
        "What are the best movies of 2024?",
        "I love watching old movies from the 1950s.",
        "Can you recommend some movies for a movie night?",
    ],
}

# Neutral messages that don't mention specific topics
NEUTRAL_MESSAGES = [
    "What should I have for dinner tonight?",
    "Can you help me write a professional email?",
    "Explain the concept of compound interest.",
    "What are some good habits to develop?",
    "Tell me an interesting fact I probably don't know.",
    "How do I organize my closet efficiently?",
    "What are the benefits of meditation?",
    "Can you help me plan my weekly schedule?",
    "What's the difference between a meteor and a meteorite?",
    "How does photosynthesis work?",
    "What are some creative gift ideas?",
    "Can you explain how bridges are engineered?",
    "What makes a good leader?",
    "How do airplanes stay in the air?",
    "What are some strategies for better sleep?",
]

QUESTION_MESSAGES = [
    "What is the capital of France?",
    "How do computers process information?",
    "Why is the sky blue?",
    "What causes earthquakes?",
    "How does the internet work?",
]

STATEMENT_MESSAGES = [
    "I went to the store yesterday and bought some groceries.",
    "The meeting has been rescheduled to next Thursday at 3pm.",
    "I finished reading that book you recommended last week.",
    "My neighbor just adopted a new kitten from the shelter.",
    "The conference was really insightful this year.",
]

LONG_MESSAGES = [
    "I've been thinking a lot lately about how to improve my daily routine and I was wondering if you could give me some detailed advice on morning habits, evening wind-down practices, and how to structure my work breaks throughout the day to maximize productivity.",
    "There are so many options for learning new skills online these days and I'm feeling overwhelmed by the choices available to me so could you help me narrow down what would be most valuable to focus on given that I work in marketing and want to transition to data analysis.",
    "My family is planning a big reunion this summer and we need to coordinate accommodations, activities, meals, and transportation for about thirty people of various ages ranging from toddlers to grandparents, and I'd love some organizational suggestions for making it run smoothly.",
]

SHORT_MESSAGES = [
    "Hi there.",
    "Thanks!",
    "OK, got it.",
    "Sure, sounds good.",
    "Tell me more.",
]


@dataclass
class TestCase:
    """A single test case for the if-then benchmark."""
    case_id: str
    condition_type: str  # content, format, counting, negation
    num_rules: int
    system_prompt: str
    user_message: str
    # For each rule: should it trigger?
    expected_triggers: list  # list of bool
    # For each rule: what marker to check for
    markers: list  # list of str
    # For verification
    verification_notes: str


def make_content_rule(topic: str, marker: str) -> dict:
    """If the user mentions [topic], include [marker] in your response."""
    return {
        "type": "content",
        "rule_text": f'If the user mentions "{topic}" in their message, you MUST include the exact text "{marker}" somewhere in your response.',
        "topic": topic,
        "marker": marker,
    }


def make_format_rule(marker: str) -> dict:
    """If the user asks a question, include [marker]."""
    return {
        "type": "format",
        "rule_text": f'If the user\'s message is a question (contains a question mark), you MUST include the exact text "{marker}" somewhere in your response.',
        "marker": marker,
    }


def make_counting_rule(marker: str, threshold: int = 20) -> dict:
    """If the user's message has more than N words, include [marker]."""
    return {
        "type": "counting",
        "rule_text": f'If the user\'s message contains more than {threshold} words, you MUST include the exact text "{marker}" somewhere in your response.',
        "marker": marker,
        "threshold": threshold,
    }


def make_negation_rule(topic: str, marker: str) -> dict:
    """If the user does NOT mention [topic], include [marker]."""
    return {
        "type": "negation",
        "rule_text": f'If the user\'s message does NOT mention "{topic}", you MUST include the exact text "{marker}" somewhere in your response.',
        "topic": topic,
        "marker": marker,
    }


def build_system_prompt(rules: list[dict]) -> str:
    """Build a system prompt from a list of rules."""
    rules_text = "\n".join(f"Rule {i+1}: {r['rule_text']}" for i, r in enumerate(rules))
    return (
        "You are a helpful assistant. You must follow ALL of the rules below precisely.\n\n"
        f"{rules_text}\n\n"
        "Important: Follow the rules exactly as stated. If a rule's condition is met, "
        "you MUST perform the specified action. If a rule's condition is NOT met, "
        "do NOT perform the action. Respond naturally to the user's message while "
        "following all applicable rules."
    )


def should_trigger_content(msg: str, topic: str) -> bool:
    return topic.lower() in msg.lower()


def should_trigger_format(msg: str) -> bool:
    return "?" in msg


def should_trigger_counting(msg: str, threshold: int = 20) -> bool:
    return len(msg.split()) > threshold


def should_trigger_negation(msg: str, topic: str) -> bool:
    return topic.lower() not in msg.lower()


def generate_single_rule_cases() -> list[TestCase]:
    """Generate test cases with a single if-then rule."""
    cases = []
    case_num = 0

    # ── Content-trigger cases ──
    for topic_name, topic_word in TOPICS[:5]:
        marker = MARKER_PHRASES[0]
        rule = make_content_rule(topic_word, marker)
        # Trigger case
        for msg in TRIGGER_MESSAGES[topic_word][:2]:
            cases.append(TestCase(
                case_id=f"content_trigger_{case_num}",
                condition_type="content",
                num_rules=1,
                system_prompt=build_system_prompt([rule]),
                user_message=msg,
                expected_triggers=[True],
                markers=[marker],
                verification_notes=f"Should trigger: message mentions '{topic_word}'",
            ))
            case_num += 1
        # No-trigger case
        for msg in random.sample(NEUTRAL_MESSAGES, 2):
            cases.append(TestCase(
                case_id=f"content_notrigger_{case_num}",
                condition_type="content",
                num_rules=1,
                system_prompt=build_system_prompt([rule]),
                user_message=msg,
                expected_triggers=[False],
                markers=[marker],
                verification_notes=f"Should NOT trigger: message doesn't mention '{topic_word}'",
            ))
            case_num += 1

    # ── Format-trigger cases (question detection) ──
    marker = MARKER_PHRASES[1]
    rule = make_format_rule(marker)
    for msg in QUESTION_MESSAGES:
        cases.append(TestCase(
            case_id=f"format_trigger_{case_num}",
            condition_type="format",
            num_rules=1,
            system_prompt=build_system_prompt([rule]),
            user_message=msg,
            expected_triggers=[True],
            markers=[marker],
            verification_notes="Should trigger: message is a question",
        ))
        case_num += 1
    for msg in STATEMENT_MESSAGES:
        cases.append(TestCase(
            case_id=f"format_notrigger_{case_num}",
            condition_type="format",
            num_rules=1,
            system_prompt=build_system_prompt([rule]),
            user_message=msg,
            expected_triggers=[False],
            markers=[marker],
            verification_notes="Should NOT trigger: message is a statement",
        ))
        case_num += 1

    # Add more format cases with different neutral messages
    for msg in random.sample(NEUTRAL_MESSAGES, 5):
        is_question = "?" in msg
        cases.append(TestCase(
            case_id=f"format_{'trigger' if is_question else 'notrigger'}_{case_num}",
            condition_type="format",
            num_rules=1,
            system_prompt=build_system_prompt([rule]),
            user_message=msg,
            expected_triggers=[is_question],
            markers=[marker],
            verification_notes=f"{'Should' if is_question else 'Should NOT'} trigger",
        ))
        case_num += 1

    # ── Counting-trigger cases ──
    marker = MARKER_PHRASES[2]
    rule = make_counting_rule(marker, threshold=20)
    for msg in LONG_MESSAGES:
        cases.append(TestCase(
            case_id=f"counting_trigger_{case_num}",
            condition_type="counting",
            num_rules=1,
            system_prompt=build_system_prompt([rule]),
            user_message=msg,
            expected_triggers=[True],
            markers=[marker],
            verification_notes=f"Should trigger: message has {len(msg.split())} words (>20)",
        ))
        case_num += 1
    for msg in SHORT_MESSAGES:
        cases.append(TestCase(
            case_id=f"counting_notrigger_{case_num}",
            condition_type="counting",
            num_rules=1,
            system_prompt=build_system_prompt([rule]),
            user_message=msg,
            expected_triggers=[False],
            markers=[marker],
            verification_notes=f"Should NOT trigger: message has {len(msg.split())} words (<=20)",
        ))
        case_num += 1

    # Add more counting cases from other messages
    for msg in random.sample(NEUTRAL_MESSAGES, 7):
        triggered = should_trigger_counting(msg, 20)
        cases.append(TestCase(
            case_id=f"counting_{'trigger' if triggered else 'notrigger'}_{case_num}",
            condition_type="counting",
            num_rules=1,
            system_prompt=build_system_prompt([rule]),
            user_message=msg,
            expected_triggers=[triggered],
            markers=[marker],
            verification_notes=f"Message has {len(msg.split())} words",
        ))
        case_num += 1

    # ── Negation-trigger cases ──
    for topic_name, topic_word in TOPICS[:5]:
        marker = MARKER_PHRASES[3]
        rule = make_negation_rule(topic_word, marker)
        # Trigger case (topic NOT mentioned → should trigger)
        for msg in random.sample(NEUTRAL_MESSAGES, 2):
            cases.append(TestCase(
                case_id=f"negation_trigger_{case_num}",
                condition_type="negation",
                num_rules=1,
                system_prompt=build_system_prompt([rule]),
                user_message=msg,
                expected_triggers=[True],
                markers=[marker],
                verification_notes=f"Should trigger: message doesn't mention '{topic_word}'",
            ))
            case_num += 1
        # No-trigger case (topic IS mentioned → should NOT trigger)
        for msg in TRIGGER_MESSAGES[topic_word][:2]:
            cases.append(TestCase(
                case_id=f"negation_notrigger_{case_num}",
                condition_type="negation",
                num_rules=1,
                system_prompt=build_system_prompt([rule]),
                user_message=msg,
                expected_triggers=[False],
                markers=[marker],
                verification_notes=f"Should NOT trigger: message mentions '{topic_word}'",
            ))
            case_num += 1

    return cases


def generate_multi_rule_cases() -> list[TestCase]:
    """Generate test cases with 2 or 4 simultaneous rules."""
    cases = []
    case_num = 0

    # ── 2-rule cases ──
    rule_pairs = [
        # content + format
        (make_content_rule("dogs", MARKER_PHRASES[0]), make_format_rule(MARKER_PHRASES[1])),
        # content + counting
        (make_content_rule("travel", MARKER_PHRASES[0]), make_counting_rule(MARKER_PHRASES[2])),
        # format + negation
        (make_format_rule(MARKER_PHRASES[1]), make_negation_rule("cooking", MARKER_PHRASES[3])),
        # content + negation
        (make_content_rule("music", MARKER_PHRASES[0]), make_negation_rule("sports", MARKER_PHRASES[3])),
    ]

    test_messages_2rule = [
        # dogs question (triggers content:dogs + format:question)
        "What are the best breeds of dogs for apartment living?",
        # dogs statement (triggers content:dogs, not format:question)
        "I just adopted two new dogs from the shelter last week.",
        # neutral question (no content trigger, triggers format:question)
        "What time does the library close today?",
        # neutral statement (no triggers for content/format)
        "I went for a walk in the park this morning.",
        # long message about travel (triggers content:travel + counting)
        "I'm planning an exciting travel adventure through Southeast Asia next year and I want to visit at least five countries including Thailand Vietnam Cambodia Laos and Myanmar over the course of three months.",
        # short neutral (no triggers)
        "Sounds great, thanks!",
        # music question without sports (triggers content:music, format:question, negation:sports)
        "What kind of music do you recommend for studying?",
        # sports statement (triggers content if applicable, blocks negation:sports)
        "I've been watching a lot of sports on TV lately.",
    ]

    for pair_idx, (rule_a, rule_b) in enumerate(rule_pairs):
        for msg in test_messages_2rule:
            # Determine triggers
            triggers = []
            markers = []
            for rule in [rule_a, rule_b]:
                m = rule["marker"]
                markers.append(m)
                if rule["type"] == "content":
                    triggers.append(should_trigger_content(msg, rule["topic"]))
                elif rule["type"] == "format":
                    triggers.append(should_trigger_format(msg))
                elif rule["type"] == "counting":
                    triggers.append(should_trigger_counting(msg, rule.get("threshold", 20)))
                elif rule["type"] == "negation":
                    triggers.append(should_trigger_negation(msg, rule["topic"]))

            cases.append(TestCase(
                case_id=f"multi2_{pair_idx}_{case_num}",
                condition_type="multi_2",
                num_rules=2,
                system_prompt=build_system_prompt([rule_a, rule_b]),
                user_message=msg,
                expected_triggers=triggers,
                markers=markers,
                verification_notes=f"2 rules: {rule_a['type']}+{rule_b['type']}",
            ))
            case_num += 1

    # ── 4-rule cases ──
    four_rules = [
        make_content_rule("dogs", MARKER_PHRASES[0]),
        make_format_rule(MARKER_PHRASES[1]),
        make_counting_rule(MARKER_PHRASES[2]),
        make_negation_rule("cooking", MARKER_PHRASES[3]),
    ]

    test_messages_4rule = [
        # dogs + question + long + no cooking → all 4 trigger
        "I've been thinking about getting some dogs and I have a lot of questions about which breeds would be the best fit for my family given that we live in a small apartment with two young children who are very active and energetic?",
        # dogs + statement + short + no cooking → content triggers, others vary
        "I love dogs.",
        # neutral question + short → format triggers, negation triggers
        "Why is the sky blue?",
        # long neutral statement + no cooking → counting + negation trigger
        "I recently finished a comprehensive renovation of my entire house which took about six months and involved replacing all the flooring installing new kitchen cabinets and completely redoing both bathrooms from scratch.",
        # short statement about cooking → nothing should trigger (no dogs, no question, short, mentions cooking)
        "I enjoy cooking dinner.",
        # dogs + question + long + cooking → content + format + counting trigger, negation does NOT
        "I want to ask about keeping dogs healthy through proper nutrition and cooking homemade meals for them because I've read several articles suggesting that commercial dog food may not be the best option for all breeds especially those with sensitive stomachs?",
    ]

    for msg in test_messages_4rule:
        triggers = []
        markers = []
        for rule in four_rules:
            m = rule["marker"]
            markers.append(m)
            if rule["type"] == "content":
                triggers.append(should_trigger_content(msg, rule["topic"]))
            elif rule["type"] == "format":
                triggers.append(should_trigger_format(msg))
            elif rule["type"] == "counting":
                triggers.append(should_trigger_counting(msg, rule.get("threshold", 20)))
            elif rule["type"] == "negation":
                triggers.append(should_trigger_negation(msg, rule["topic"]))

        cases.append(TestCase(
            case_id=f"multi4_{case_num}",
            condition_type="multi_4",
            num_rules=4,
            system_prompt=build_system_prompt(four_rules),
            user_message=msg,
            expected_triggers=triggers,
            markers=markers,
            verification_notes=f"4 rules: content+format+counting+negation, triggers={triggers}",
        ))
        case_num += 1

    return cases


def generate_unconditional_baseline_cases() -> list[TestCase]:
    """Baseline: unconditional instructions (always include marker).
    Tests action execution ability without conditional logic."""
    cases = []
    marker = MARKER_PHRASES[0]
    system_prompt = (
        f"You are a helpful assistant. You MUST always include the exact text "
        f'"{marker}" somewhere in every response you give, no matter what the user says.'
    )
    for i, msg in enumerate(random.sample(NEUTRAL_MESSAGES + SHORT_MESSAGES, 10)):
        cases.append(TestCase(
            case_id=f"unconditional_{i}",
            condition_type="unconditional",
            num_rules=1,
            system_prompt=system_prompt,
            user_message=msg,
            expected_triggers=[True],
            markers=[marker],
            verification_notes="Unconditional baseline: marker should always appear",
        ))
    return cases


def generate_all_cases() -> list[TestCase]:
    """Generate the complete benchmark."""
    cases = []
    cases.extend(generate_single_rule_cases())
    cases.extend(generate_multi_rule_cases())
    cases.extend(generate_unconditional_baseline_cases())
    return cases


def verify_response(response: str, test_case: TestCase) -> dict:
    """Verify whether the model's response correctly follows the if-then rules.

    Returns a dict with per-rule results and overall accuracy.
    """
    results = []
    for i, (expected, marker) in enumerate(zip(test_case.expected_triggers, test_case.markers)):
        marker_present = marker in response
        correct = (marker_present == expected)

        if expected and not marker_present:
            error_type = "false_negative"  # Should have triggered but didn't
        elif not expected and marker_present:
            error_type = "false_positive"  # Shouldn't have triggered but did
        else:
            error_type = None

        results.append({
            "rule_index": i,
            "marker": marker,
            "expected_trigger": expected,
            "marker_present": marker_present,
            "correct": correct,
            "error_type": error_type,
        })

    all_correct = all(r["correct"] for r in results)
    return {
        "case_id": test_case.case_id,
        "condition_type": test_case.condition_type,
        "num_rules": test_case.num_rules,
        "per_rule_results": results,
        "all_rules_correct": all_correct,
    }


if __name__ == "__main__":
    cases = generate_all_cases()
    print(f"Total test cases generated: {len(cases)}")

    # Count by type
    from collections import Counter
    type_counts = Counter(c.condition_type for c in cases)
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")

    # Count trigger vs no-trigger
    trigger_counts = Counter()
    for c in cases:
        for exp in c.expected_triggers:
            trigger_counts[exp] += 1
    print(f"\nExpected triggers: True={trigger_counts[True]}, False={trigger_counts[False]}")

    # Save benchmark
    benchmark_data = [asdict(c) for c in cases]
    with open("results/benchmark.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nBenchmark saved to results/benchmark.json")
