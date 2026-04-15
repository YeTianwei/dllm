"""
Run:
    CUDA_VISIBLE_DEVICES=3 python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py \
        --method baseline_llada \
        --model_name_or_path /data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final
"""

from __future__ import annotations

import dllm
from dllm.acceleration.base import AccelerationMethod
from dllm.acceleration.benchmark import BenchmarkConfig


class BaselineLLaDAMethod(AccelerationMethod):
    method_name = "baseline_llada"

    def build_model(self, config: BenchmarkConfig):
        model_args = dllm.utils.ModelArguments(
            model_name_or_path=config.model_name_or_path
        )
        return dllm.utils.get_model(model_args=model_args).eval()

    def build_tokenizer(self, config: BenchmarkConfig):
        model_args = dllm.utils.ModelArguments(
            model_name_or_path=config.model_name_or_path
        )
        return dllm.utils.get_tokenizer(model_args=model_args)

    def build_sampler(self, *, model, tokenizer, config: BenchmarkConfig):
        return dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
