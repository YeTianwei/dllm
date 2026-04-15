"""
Run:
    python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py --help
"""

from .benchmark import BenchmarkConfig, BenchmarkResult, run_benchmark
from .registry import get_method, list_methods, register_method
from .prompts import get_prompt_set, list_prompt_sets

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "get_prompt_set",
    "get_method",
    "list_prompt_sets",
    "list_methods",
    "register_method",
    "run_benchmark",
]
