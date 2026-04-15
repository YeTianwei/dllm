"""
Run:
    CUDA_VISIBLE_DEVICES=3 python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py \
        --model_name_or_path /data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final \
        --output_json /data/ytw/VLA_baseline/dllm/.artifacts/benchmark_smoke.json
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import dllm
from dllm.acceleration.prompts import PromptExample


@dataclass
class BenchmarkConfig:
    model_name_or_path: str = ""
    method: str = "baseline_llada"
    batch_size: int = 1
    num_repeats: int = 3
    warmup_runs: int = 1
    max_new_tokens: int = 64
    steps: int = 64
    block_size: int = 64
    prompt_set: str = "llada_smoke"
    output_json: str | None = None


@dataclass
class BenchmarkResult:
    method: str
    model_name_or_path: str
    sampler_name: str
    config: dict[str, Any]
    num_prompts: int
    mean_latency_sec: float | None
    p50_latency_sec: float | None
    p95_latency_sec: float | None
    tokens_per_sec: float | None
    samples_per_sec: float | None
    peak_gpu_mem_mb: float | None
    generated_texts: list[str]
    trimmed_outputs: list[str]
    success: bool
    error_message: str | None = None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def _batch_messages(examples: list[PromptExample], batch_size: int) -> list[list[PromptExample]]:
    return [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]


def _prepare_inputs(
    tokenizer,
    examples: list[PromptExample],
) -> list[list[int]]:
    messages = [[{"role": "user", "content": example.prompt}] for example in examples]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )


def _run_once(
    sampler,
    tokenizer,
    examples: list[PromptExample],
    config: BenchmarkConfig,
) -> tuple[float, int, float | None, list[str], list[str]]:
    generated_texts: list[str] = []
    trimmed_outputs: list[str] = []
    total_generated_tokens = 0
    peak_gpu_mem_mb: float | None = None

    total_elapsed = 0.0

    for batch_examples in _batch_messages(examples, config.batch_size):
        inputs = _prepare_inputs(tokenizer, batch_examples)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = sampler.sample(
            inputs=inputs,
            return_dict=True,
            max_new_tokens=config.max_new_tokens,
            steps=config.steps,
            block_size=config.block_size,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed

        sequences = outputs.sequences.tolist()
        batch_generated = dllm.utils.sample_trim(tokenizer, sequences, inputs)
        batch_full = [tokenizer.decode(seq) for seq in sequences]
        generated_texts.extend(batch_full)
        trimmed_outputs.extend(batch_generated)

        for seq, prompt in zip(sequences, inputs):
            total_generated_tokens += max(0, len(seq) - len(prompt))

        if torch.cuda.is_available():
            current_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_gpu_mem_mb = (
                current_peak_mb
                if peak_gpu_mem_mb is None
                else max(peak_gpu_mem_mb, current_peak_mb)
            )

    return (
        total_elapsed,
        total_generated_tokens,
        peak_gpu_mem_mb,
        generated_texts,
        trimmed_outputs,
    )


def run_benchmark(
    method_name: str,
    method,
    prompt_examples: list[PromptExample],
    config: BenchmarkConfig,
) -> BenchmarkResult:
    try:
        model = method.build_model(config)
        tokenizer = method.build_tokenizer(config)
        sampler = method.build_sampler(model=model, tokenizer=tokenizer, config=config)

        for _ in range(config.warmup_runs):
            _run_once(sampler, tokenizer, prompt_examples, config)

        latencies: list[float] = []
        generated_token_counts = 0
        peak_gpu_mem_mb: float | None = None
        last_generated_texts: list[str] = []
        last_trimmed_outputs: list[str] = []

        for _ in range(config.num_repeats):
            elapsed, gen_tokens, peak_mb, generated_texts, trimmed_outputs = _run_once(
                sampler, tokenizer, prompt_examples, config
            )
            latencies.append(elapsed)
            generated_token_counts += gen_tokens
            if peak_mb is not None:
                peak_gpu_mem_mb = (
                    peak_mb if peak_gpu_mem_mb is None else max(peak_gpu_mem_mb, peak_mb)
                )
            last_generated_texts = generated_texts
            last_trimmed_outputs = trimmed_outputs

        total_latency = sum(latencies)
        num_samples = len(prompt_examples) * max(config.num_repeats, 1)
        result = BenchmarkResult(
            method=method_name,
            model_name_or_path=config.model_name_or_path,
            sampler_name=type(sampler).__name__,
            config=asdict(config),
            num_prompts=len(prompt_examples),
            mean_latency_sec=(statistics.mean(latencies) if latencies else None),
            p50_latency_sec=_percentile(latencies, 50),
            p95_latency_sec=_percentile(latencies, 95),
            tokens_per_sec=(
                generated_token_counts / total_latency if total_latency > 0 else None
            ),
            samples_per_sec=(num_samples / total_latency if total_latency > 0 else None),
            peak_gpu_mem_mb=peak_gpu_mem_mb,
            generated_texts=last_generated_texts,
            trimmed_outputs=last_trimmed_outputs,
            success=True,
        )
    except Exception as exc:
        result = BenchmarkResult(
            method=method_name,
            model_name_or_path=config.model_name_or_path,
            sampler_name="",
            config=asdict(config),
            num_prompts=len(prompt_examples),
            mean_latency_sec=None,
            p50_latency_sec=None,
            p95_latency_sec=None,
            tokens_per_sec=None,
            samples_per_sec=None,
            peak_gpu_mem_mb=None,
            generated_texts=[],
            trimmed_outputs=[],
            success=False,
            error_message=str(exc),
        )

    if config.output_json:
        output_path = Path(config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return result
