"""
Run:
    python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py --list_methods True
"""

from __future__ import annotations

from dllm.acceleration.base import AccelerationMethod

_METHODS: dict[str, AccelerationMethod] = {}


def register_method(method: AccelerationMethod) -> AccelerationMethod:
    if not method.method_name:
        raise ValueError("AccelerationMethod.method_name must be non-empty.")
    _METHODS[method.method_name] = method
    return method


def get_method(name: str) -> AccelerationMethod:
    if name not in _METHODS:
        supported = ", ".join(list_methods())
        raise ValueError(f"Unknown acceleration method {name!r}. Supported: {supported}")
    return _METHODS[name]


def list_methods() -> list[str]:
    return sorted(_METHODS.keys())


from dllm.acceleration.methods.baseline_llada import BaselineLLaDAMethod
from dllm.acceleration.methods.sdtt_llada import SDTTLLaDAMethod

register_method(BaselineLLaDAMethod())
register_method(SDTTLLaDAMethod())
