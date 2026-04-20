"""
Run:
    python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py --list_prompt_sets True
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptExample:
    category: str
    prompt: str


_PROMPT_SETS: dict[str, list[PromptExample]] = {
    "llada_smoke": [
        PromptExample(
            category="reasoning",
            prompt="If a train moves at 60 km/h for 3 hours, how far does it travel?",
        ),
        PromptExample(
            category="coding",
            prompt="Write a short Python function that checks whether a string is a palindrome.",
        ),
        PromptExample(
            category="generic",
            prompt="Write a concise paragraph about why clean code matters in research projects.",
        ),
    ],
    "llada_reasoning": [
        PromptExample(
            category="reasoning",
            prompt="Lily saves 12 dollars per week for 8 weeks. How much money does she save in total?",
        ),
        PromptExample(
            category="reasoning",
            prompt="A rectangle has length 9 and width 4. What is its area?",
        ),
    ],
    "llada_coding": [
        PromptExample(
            category="coding",
            prompt="Write a Python function that merges two sorted lists into one sorted list.",
        ),
        PromptExample(
            category="coding",
            prompt="Explain how to reverse a list in Python and show a simple example.",
        ),
    ],
    "llada_eval_large": [
        PromptExample(
            category="reasoning",
            prompt="Lily saves 12 dollars per week for 8 weeks. How much money does she save in total?",
        ),
        PromptExample(
            category="reasoning",
            prompt="A rectangle has length 9 and width 4. What is its area?",
        ),
        PromptExample(
            category="reasoning",
            prompt="A store sells notebooks for 3 dollars each. If Sam buys 7 notebooks, how much does he pay in total?",
        ),
        PromptExample(
            category="reasoning",
            prompt="If a car travels 45 miles per hour for 4 hours, how far does it travel?",
        ),
        PromptExample(
            category="reasoning",
            prompt="A box contains 24 apples. If you divide them equally among 6 people, how many apples does each person get?",
        ),
        PromptExample(
            category="reasoning",
            prompt="What is the perimeter of a rectangle with length 10 and width 3?",
        ),
        PromptExample(
            category="coding",
            prompt="Write a short Python function that checks whether a string is a palindrome.",
        ),
        PromptExample(
            category="coding",
            prompt="Write a Python function that returns the factorial of a non-negative integer.",
        ),
        PromptExample(
            category="coding",
            prompt="Write a Python function that counts how many times each word appears in a sentence.",
        ),
        PromptExample(
            category="coding",
            prompt="Explain how to remove duplicates from a Python list while preserving order, and show a simple example.",
        ),
        PromptExample(
            category="coding",
            prompt="Write a Python function that finds the maximum value in a list without using the built-in max function.",
        ),
        PromptExample(
            category="generic",
            prompt="Write a concise paragraph about why clean code matters in research projects.",
        ),
        PromptExample(
            category="generic",
            prompt="Write a concise paragraph explaining why reproducibility is important in machine learning experiments.",
        ),
        PromptExample(
            category="generic",
            prompt="Summarize the benefits of writing clear experiment logs in a research codebase.",
        ),
        PromptExample(
            category="generic",
            prompt="Explain in one short paragraph how code review can improve the quality of a research project.",
        ),
        PromptExample(
            category="generic",
            prompt="Write a short explanation of why small benchmark suites are useful before running large experiments.",
        ),
    ],
}


def list_prompt_sets() -> list[str]:
    return sorted(_PROMPT_SETS.keys())


def get_prompt_set(name: str) -> list[PromptExample]:
    if name not in _PROMPT_SETS:
        supported = ", ".join(list_prompt_sets())
        raise ValueError(f"Unknown prompt_set={name!r}. Supported: {supported}")
    return _PROMPT_SETS[name]
