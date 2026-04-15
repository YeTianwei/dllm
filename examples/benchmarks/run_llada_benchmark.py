"""
Run:
    CUDA_VISIBLE_DEVICES=3 python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py \
        --model_name_or_path /data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final \
        --output_json /data/ytw/VLA_baseline/dllm/.artifacts/benchmark_smoke.json
"""

from dataclasses import dataclass
import json

import transformers

import dllm
from dllm.acceleration.benchmark import BenchmarkConfig, run_benchmark
from dllm.acceleration.prompts import get_prompt_set, list_prompt_sets


@dataclass
class ScriptArguments(BenchmarkConfig):
    list_prompt_sets: bool = False


class _DirectLLaDAMethod:
    method_name = "baseline_llada"

    @staticmethod
    def build_model(config: BenchmarkConfig):
        model_args = dllm.utils.ModelArguments(model_name_or_path=config.model_name_or_path)
        return dllm.utils.get_model(model_args=model_args).eval()

    @staticmethod
    def build_tokenizer(config: BenchmarkConfig):
        model_args = dllm.utils.ModelArguments(model_name_or_path=config.model_name_or_path)
        return dllm.utils.get_tokenizer(model_args=model_args)

    @staticmethod
    def build_sampler(model, tokenizer, config: BenchmarkConfig):
        return dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)


def main():
    parser = transformers.HfArgumentParser((ScriptArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    if args.list_prompt_sets:
        for name in list_prompt_sets():
            print(name)
        return

    prompts = get_prompt_set(args.prompt_set)
    result = run_benchmark(
        method_name=args.method,
        method=_DirectLLaDAMethod(),
        prompt_examples=prompts,
        config=BenchmarkConfig(
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
        ),
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
