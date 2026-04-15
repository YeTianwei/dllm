# Benchmarks

This directory contains benchmark entrypoints and checked-in reference results for
the acceleration platform work.

## Environment

Use the validated environment and keep benchmarks on a single GPU:

```bash
cd /data/ytw/VLA_baseline/dllm
source ~/.zshrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/timer/miniconda3/envs/dllm
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=3
```

`PYTHONNOUSERSITE=1` is important on this machine because user-site packages can
override the conda environment and break imports such as `pandas` and `datasets`.

## Baseline LLaDA benchmark

Reference command:

```bash
python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py \
  --method baseline_llada \
  --model_name_or_path /data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final \
  --output_json /data/ytw/VLA_baseline/dllm/.artifacts/llada_baseline_smoke.json
```

Reference result:

- `/data/ytw/VLA_baseline/dllm/examples/benchmarks/results/llada_baseline_smoke.json`

## SDTT-style LLaDA distillation

Train a trajectory-distilled student from the local smoke checkpoint:

```bash
python /data/ytw/VLA_baseline/dllm/examples/benchmarks/train_sdtt_llada.py \
  --teacher_model_name_or_path /data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final \
  --student_model_name_or_path /data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final \
  --dataset_args "tatsu-lab/alpaca[train:8,test:4]" \
  --max_steps 1 \
  --teacher_steps 16 \
  --student_steps 4 \
  --block_size 4 \
  --output_dir /data/ytw/VLA_baseline/dllm/.models/sdtt-llada-smoke
```

Benchmark the distilled checkpoint:

```bash
python /data/ytw/VLA_baseline/dllm/examples/benchmarks/run_llada_benchmark.py \
  --method sdtt_llada \
  --model_name_or_path /data/ytw/VLA_baseline/dllm/.models/sdtt-llada-smoke/checkpoint-final \
  --output_json /data/ytw/VLA_baseline/dllm/.artifacts/sdtt_llada_smoke.json
```

The `sdtt_llada` method reads `/data/ytw/VLA_baseline/dllm/.models/.../sdtt_config.json`
and automatically applies the stored `student_steps` and `block_size` unless you
override them on the CLI.

## LSD Status

`/data/ytw/VLA_baseline/LSD` is currently treated as a planned method only.
Its public repository does not ship a complete training and evaluation pipeline,
so it is not wired into the benchmark platform in this version.
