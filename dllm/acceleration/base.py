"""
Run:
    python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py --help
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizer

from dllm.acceleration.benchmark import BenchmarkConfig
from dllm.core.samplers import BaseSampler


@dataclass
class AccelerationMethodConfig:
    method_name: str
    extra_kwargs: dict[str, Any] | None = None


class AccelerationMethod:
    method_name: str = ""

    def prepare_benchmark_config(self, config: BenchmarkConfig) -> BenchmarkConfig:
        return config

    def build_model(self, config: BenchmarkConfig) -> PreTrainedModel:
        raise NotImplementedError

    def build_tokenizer(self, config: BenchmarkConfig) -> PreTrainedTokenizer:
        raise NotImplementedError

    def build_sampler(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: BenchmarkConfig,
    ) -> BaseSampler:
        raise NotImplementedError
