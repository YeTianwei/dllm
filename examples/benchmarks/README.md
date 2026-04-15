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
