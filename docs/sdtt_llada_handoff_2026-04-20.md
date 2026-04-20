# SDTT LLaDA Handoff

Date: 2026-04-20

## Objective

Continue the SDTT-LLaDA experiment sequence with the goal:

`Improve generation speed while preserving output quality as much as possible.`

## Repository State

The following source files were modified in this workstream:

- `dllm/acceleration/methods/sdtt_llada.py`
- `dllm/acceleration/prompts.py`
- `dllm/utils/models.py`
- `examples/benchmarks/sdtt_llada/run_experiment.sh`
- `examples/benchmarks/train_sdtt_llada.py`
- `scripts/accelerate_configs/zero3_offload.yaml`
- `scripts/accelerate_configs/zero3_param_offload.yaml`

Two documentation files were also added:

- `docs/sdtt_llada_pilot_report_2026-04-20.md`
- `docs/sdtt_llada_handoff_2026-04-20.md`

## Important Functional Changes

### Multi-GPU Training

`examples/benchmarks/sdtt_llada/run_experiment.sh` now launches multi-GPU training through `accelerate launch` instead of plain single-process `python`.

### Safe ZeRO-3 Loading

`examples/benchmarks/train_sdtt_llada.py` now:

- avoids the old `gradient_checkpointing` argument parser conflict,
- handles teacher and student model loading differently under ZeRO-3,
- temporarily disables the Hugging Face DeepSpeed global config while loading the frozen teacher,
- uses `bfloat16` by default rather than `float32`.

### Local-Cache-Aware Model Loading

`dllm/acceleration/methods/sdtt_llada.py` now sets `local_files_only=True` automatically when the model can be resolved from the local Hugging Face cache.

This was added because the LoRA checkpoint at:

- `/data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final`
- `.models/smoke_test_llada_sft/checkpoint-final`

points to base model:

- `GSAI-ML/LLaDA-8B-Base`

and the benchmark/training path was previously stalling on Hugging Face metadata requests despite the base model already being cached locally.

### Benchmark Defaults

`dllm/acceleration/methods/sdtt_llada.py` now defaults SDTT benchmark inference to:

- `steps=24`
- `block_size=24`

when benchmarking a final SDTT checkpoint that otherwise would have used the original `8/8` metadata.

`examples/benchmarks/sdtt_llada/run_experiment.sh` also explicitly passes `--steps 24 --block_size 24` for the student benchmark stage.

### New Prompt Set

A new evaluation set was added:

- `llada_eval_large`

Defined in:

- `dllm/acceleration/prompts.py`

It contains 16 prompts spanning reasoning, coding, and generic writing.

## Environment Constraints

This machine has a CUDA mismatch for DeepSpeed CPUAdam:

- PyTorch CUDA: `12.1`
- Installed CUDA toolkit detected by DeepSpeed: `13.1`

Consequences:

- `scripts/accelerate_configs/zero3_offload.yaml` is not usable on this machine.
- Any config that forces DeepSpeed CPUAdam compilation will fail.

Use this config instead:

- `scripts/accelerate_configs/zero3_param_offload.yaml`

This offloads parameters to CPU but keeps optimizer offload disabled, avoiding CPUAdam.

## Validated Commands

### 1. Original Pilot Training Recipe

This completed successfully on the current machine:

```bash
source ~/.zshrc
source /home/timer/miniconda3/etc/profile.d/conda.sh
conda activate /home/timer/miniconda3/envs/dllm

bash examples/benchmarks/sdtt_llada/run_experiment.sh \
  --stage train \
  --preset pilot \
  --cuda_device 0,1 \
  --num_gpus 2 \
  --accelerate_config scripts/accelerate_configs/zero3_param_offload.yaml \
  --load_in_4bit true \
  --gradient_checkpointing false
```

Output checkpoint:

- `.models/sdtt-llada-pilot/checkpoint-final`

### 2. Larger Benchmark

```bash
source ~/.zshrc
source /home/timer/miniconda3/etc/profile.d/conda.sh
conda activate /home/timer/miniconda3/envs/dllm

bash examples/benchmarks/sdtt_llada/run_experiment.sh \
  --stage benchmark \
  --preset pilot \
  --cuda_device 0 \
  --prompt_set llada_eval_large
```

Outputs:

- `.artifacts/sdtt_llada/pilot/baseline.json`
- `.artifacts/sdtt_llada/pilot/student.json`

### 3. Reasoning Scan

Files already generated:

- `.artifacts/sdtt_llada/scan_reasoning/baseline_64.json`
- `.artifacts/sdtt_llada/scan_reasoning/student_8.json`
- `.artifacts/sdtt_llada/scan_reasoning/student_12.json`
- `.artifacts/sdtt_llada/scan_reasoning/student_16.json`
- `.artifacts/sdtt_llada/scan_reasoning/student_24.json`
- `.artifacts/sdtt_llada/scan_reasoning/student_32.json`

Best currently validated tradeoff for the existing pilot checkpoint:

- `24/24`

### 4. Pending 16/16 Retraining Command

This is the next recommended training run:

```bash
source ~/.zshrc
source /home/timer/miniconda3/etc/profile.d/conda.sh
conda activate /home/timer/miniconda3/envs/dllm

bash examples/benchmarks/sdtt_llada/run_experiment.sh \
  --stage train \
  --preset pilot \
  --cuda_device 0,1 \
  --num_gpus 2 \
  --accelerate_config scripts/accelerate_configs/zero3_param_offload.yaml \
  --load_in_4bit true \
  --gradient_checkpointing false \
  --student_steps 16 \
  --block_size 16 \
  --output_tag 16x16
```

Expected outputs:

- `.models/sdtt-llada-pilot-16x16`
- `.artifacts/sdtt_llada/pilot-16x16`

## Known Experimental Conclusions

### Current Pilot Student (`8-step` training target)

Good:

- Large speedups are real.
- Benchmarking pipeline is healthy.
- Short prompts are often acceptable.

Bad:

- Quality degrades badly at aggressive inference settings like `8/8`.
- Even with benchmark-time recovery to `24/24`, quality still drops on longer generic and coding prompts.

### Current Best Benchmark-Side Tradeoff

For the existing pilot checkpoint, `24/24` is the best currently validated setting.

This does not fully preserve quality, but it is much better than `8/8` and still gives about `2.7x` speedup on the larger prompt set.

## Recommended Next Work for Another Agent

1. Run the `16/16` training job on the new machine.
2. Benchmark the new checkpoint on:
   - `llada_reasoning`
   - `llada_eval_large`
3. Compare these four items:
   - baseline `64/64`
   - current pilot checkpoint benchmarked at `24/24`
   - new `16/16` checkpoint benchmarked at its default setting
   - new `16/16` checkpoint benchmarked at a small scan around `16/20/24`
4. Decide whether:
   - `16/16` is already the desired speed/quality tradeoff,
   - or a longer quality-oriented training run is needed.

## Things To Avoid

- Do not use `scripts/accelerate_configs/zero3_offload.yaml` on the current machine.
- Do not assume `8-step` student metadata is a good benchmark default; use `24/24` unless explicitly experimenting.
- Do not overwrite `.models/sdtt-llada-pilot`; use `--output_tag` for new experiments.
