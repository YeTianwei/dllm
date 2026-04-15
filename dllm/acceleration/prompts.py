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
}


def list_prompt_sets() -> list[str]:
    return sorted(_PROMPT_SETS.keys())


def get_prompt_set(name: str) -> list[PromptExample]:
    if name not in _PROMPT_SETS:
        supported = ", ".join(list_prompt_sets())
        raise ValueError(f"Unknown prompt_set={name!r}. Supported: {supported}")
    return _PROMPT_SETS[name]
