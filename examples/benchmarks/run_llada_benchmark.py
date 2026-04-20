"""
Run:
    CUDA_VISIBLE_DEVICES=3 python examples/benchmarks/run_llada_benchmark.py \
        --model_name_or_path .models/smoke_test_llada_sft/checkpoint-final \
        --output_json .artifacts/benchmark_smoke.json
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path

import transformers

from dllm.acceleration.benchmark import BenchmarkConfig, run_benchmark
from dllm.acceleration.prompts import get_prompt_set, list_prompt_sets
from dllm.acceleration.registry import get_method, list_methods


@dataclass
class ScriptArguments(BenchmarkConfig):
    list_prompt_sets: bool = False
    list_methods: bool = False


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return str((REPO_ROOT / path).resolve())


def main():
    parser = transformers.HfArgumentParser((ScriptArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    if args.list_prompt_sets:
        for name in list_prompt_sets():
            print(name)
        return
    if args.list_methods:
        for name in list_methods():
            print(name)
        return
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless using list commands.")
    args.model_name_or_path = _resolve_repo_path(args.model_name_or_path)
    args.output_json = _resolve_repo_path(args.output_json)

    prompts = get_prompt_set(args.prompt_set)
    method = get_method(args.method)
    benchmark_config = BenchmarkConfig(
        method=args.method,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        num_repeats=args.num_repeats,
        warmup_runs=args.warmup_runs,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_size=args.block_size,
        prompt_set=args.prompt_set,
        output_json=args.output_json,
    )
    benchmark_config = method.prepare_benchmark_config(benchmark_config)
    result = run_benchmark(
        method_name=args.method,
        method=method,
        prompt_examples=prompts,
        config=benchmark_config,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
