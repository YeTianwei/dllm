# SDTT LLaDA Pilot Report

Date: 2026-04-20

## Executive Summary

This round focused on getting `/data/ytw/VLA_baseline/dllm/examples/benchmarks/sdtt_llada/run_experiment.sh` and `/data/ytw/VLA_baseline/dllm/examples/benchmarks/train_sdtt_llada.py` to run SDTT-style LLaDA training and benchmarking on 2 GPUs reliably.

The main engineering outcome is positive:

- Multi-GPU training is now wired correctly through `accelerate launch`.
- The SDTT training path can run on 2x24GB GPUs with `4bit + no gradient checkpointing + ZeRO-3 param offload`.
- Benchmarking is functional and reproducible.

The main modeling outcome is mixed:

- The original aggressive student configuration (`student_steps=8`, `block_size=8`) gives a large speedup but causes major quality degradation.
- A benchmark-side quality/speed tradeoff scan showed that `steps=24`, `block_size=24` is the best currently validated inference setting for the existing pilot student checkpoint.
- On a larger 16-prompt benchmark set, the `24/24` student remains about `2.7x` faster than the baseline, but still shows noticeable quality degradation on longer generic and coding prompts.

## Code Changes Completed

The following implementation work is complete:

- `/data/ytw/VLA_baseline/dllm/examples/benchmarks/sdtt_llada/run_experiment.sh`
  - Fixed multi-GPU launch to use `accelerate launch`.
  - Added support for overriding `max_length`, `teacher_steps`, `student_steps`, `block_size`, `gradient_checkpointing`, and `output_tag`.
  - Moved override application to after preset selection so overrides are not silently clobbered.
  - Set the student benchmark command to use the validated `24/24` inference setting by default.

- `/data/ytw/VLA_baseline/dllm/examples/benchmarks/train_sdtt_llada.py`
  - Fixed DDP/ZeRO launch issues.
  - Removed the duplicate `gradient_checkpointing` argument conflict.
  - Added safer ZeRO-3 handling for teacher vs. student model loading.
  - Switched the default dtype to `bfloat16` to reduce memory pressure.

- `/data/ytw/VLA_baseline/dllm/dllm/acceleration/methods/sdtt_llada.py`
  - Added local-cache-aware model loading so cached Hugging Face models can be loaded without unnecessary network metadata calls.
  - Added a benchmark default override so SDTT student inference now defaults to `24` steps and `24` block size when benchmarked from the final checkpoint.

- `/data/ytw/VLA_baseline/dllm/dllm/utils/models.py`
  - Added explicit `device_map` override support so higher-level loading logic is not silently ignored.

- `/data/ytw/VLA_baseline/dllm/dllm/acceleration/prompts.py`
  - Added `llada_eval_large`, a 16-prompt evaluation set spanning reasoning, coding, and generic writing prompts.

- `/data/ytw/VLA_baseline/dllm/scripts/accelerate_configs/zero3_offload.yaml`
  - Added during exploration, but this config is not usable on the current machine because it triggers DeepSpeed CPUAdam compilation against a CUDA toolkit mismatch.

- `/data/ytw/VLA_baseline/dllm/scripts/accelerate_configs/zero3_param_offload.yaml`
  - Added and validated as the correct low-memory config on the current hardware.

## Key Environment Findings

- PyTorch reports CUDA `12.1`.
- DeepSpeed detects installed CUDA toolkit `13.1`.
- This mismatch breaks DeepSpeed CPUAdam compilation and makes full optimizer CPU offload unusable on the current machine.

Consequence:

- `/data/ytw/VLA_baseline/dllm/scripts/accelerate_configs/zero3_offload.yaml` is not viable on this machine.
- `/data/ytw/VLA_baseline/dllm/scripts/accelerate_configs/zero3_param_offload.yaml` is the correct workaround because it avoids CPUAdam while still reducing memory pressure.

## Validated Training Recipe

The following training recipe is the validated low-memory path for the current hardware:

- 2 GPUs
- `load_in_4bit=true`
- `gradient_checkpointing=false`
- `accelerate_config=/data/ytw/VLA_baseline/dllm/scripts/accelerate_configs/zero3_param_offload.yaml`

This recipe was used to train:

- `/data/ytw/VLA_baseline/dllm/.models/sdtt-llada-pilot`

with:

- `teacher_steps=64`
- `student_steps=8`
- `block_size=8`

Training completed successfully and produced:

- `/data/ytw/VLA_baseline/dllm/.models/sdtt-llada-pilot/checkpoint-final`

## Benchmark Results

### 1. Small Smoke Benchmark

Files:

- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/smoke/baseline.json`
- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/smoke/student.json`

Observed outcome:

- Baseline mean latency: `1.9637s`
- Student mean latency: `0.8820s`
- Student speedup: about `2.2x`
- Quality on these short prompts was broadly acceptable.

### 2. Reasoning Benchmark for Original Student

Files:

- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/pilot/baseline.json`
- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/pilot/student.json`

Observed outcome for the original `8/8` student benchmark:

- Baseline mean latency: `6.1752s`
- Student mean latency: `0.7306s`
- Student speedup: about `8.5x`
- Quality degradation was severe. Outputs showed repetition, malformed phrasing, and template pollution.

### 3. Reasoning Tradeoff Scan

Files:

- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/scan_reasoning/baseline_64.json`
- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/scan_reasoning/student_8.json`
- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/scan_reasoning/student_12.json`
- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/scan_reasoning/student_16.json`
- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/scan_reasoning/student_24.json`
- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/scan_reasoning/student_32.json`

Measured tradeoff:

| Config | Mean Latency | Speedup vs Baseline |
| --- | ---: | ---: |
| baseline `64/64` | `8.0439s` | `1.00x` |
| student `8/8` | `0.9756s` | `8.25x` |
| student `12/12` | `1.5081s` | `5.33x` |
| student `16/16` | `1.9550s` | `4.11x` |
| student `24/24` | `2.9701s` | `2.71x` |
| student `32/32` | `3.9766s` | `2.02x` |

Interpretation:

- `8/8` was too aggressive.
- `12/12` improved fluency but still made arithmetic mistakes.
- `16/16` was not reliably better than `12/12` on the sampled prompts.
- `24/24` was the best validated quality/speed tradeoff for the existing `8-step` student checkpoint.
- `32/32` was slower and did not improve enough to justify the extra cost.

### 4. Larger 16-Prompt Evaluation

Files:

- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/pilot/baseline.json`
- `/data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada/pilot/student.json`

These files were later overwritten to evaluate `llada_eval_large`.

Results:

- Baseline mean latency: `47.9006s`
- Student mean latency: `17.7142s`
- Student speedup: about `2.7x`
- Baseline tokens/sec: `21.38`
- Student tokens/sec: `57.81`

Quality summary:

- Short arithmetic prompts were much better than the original `8/8` outputs.
- Several reasoning prompts were correct or nearly correct.
- Coding prompts were partially acceptable but still showed formatting and content degradation.
- Longer generic writing prompts still exhibited repetition, awkward phrasing, or semantic degradation.

Bottom line:

- The current best validated setting (`24/24` at benchmark time) is promising but does not yet meet the bar for "speedup with quality preserved" across the full prompt set.

## Current Recommendation

The next most useful experiment is to train a new SDTT student with a less aggressive target:

- `teacher_steps=64`
- `student_steps=16`
- `block_size=16`

Rationale:

- The benchmark scan indicates the current model behaves better when allowed more denoising steps.
- Training a student directly for `16/16` is more principled than training `8/8` and then recovering quality only by increasing inference steps.

## Current Blocker

The current machine is memory constrained for these experiments. The user plans to move to a different machine with more available VRAM or a more compatible CUDA/DeepSpeed environment before continuing the next training round.

## Suggested Next Actions

1. Run the `16/16` training recipe on the new machine.
2. Benchmark the new checkpoint on:
   - `llada_reasoning`
   - `llada_eval_large`
3. Compare the new `16/16` checkpoint against:
   - baseline `64/64`
   - current pilot checkpoint benchmarked at `24/24`
4. Decide whether the new checkpoint is good enough, or whether a further quality-oriented training run is needed.
