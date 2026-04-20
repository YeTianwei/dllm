# SDTT LLaDA Experiments

This directory contains reproducible single-GPU experiment entrypoints for
`sdtt_llada`.

All commands assume:

```bash
cd <repo_root>
source ~/.zshrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/timer/miniconda3/envs/dllm
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=3
```

## Presets

The experiment runner currently provides three presets:

- `smoke`
  - Minimal sanity run for debugging the pipeline
  - Dataset: `tatsu-lab/alpaca[train:8,test:4]`
  - Teacher steps: `16`
  - Student steps: `4`
  - Max steps: `1`
  - Learning rate: `2e-5`
  - Batch size: `1`
  - Save steps: `10`

- `pilot`
  - Lightweight single-GPU research run intended for iteration
  - Dataset: `allenai/tulu-3-sft-mixture[train:1024,test:128]`
  - Teacher steps: `64`
  - Student steps: `8`
  - Max steps: `200`
  - Learning rate: `2e-5`
  - Batch size: `1`
  - Save steps: `100`

- `full`
  - Longer single-GPU run with the same method but larger budget
  - Dataset: `allenai/tulu-3-sft-mixture[train:4096,test:256]`
  - Teacher steps: `64`
  - Student steps: `8`
  - Max steps: `1000`
  - Learning rate: `2e-5`
  - Batch size: `1`
  - Save steps: `200`

All presets default to:

- Teacher checkpoint: `.models/smoke_test_llada_sft/checkpoint-final`
- Student initialization checkpoint: `.models/smoke_test_llada_sft/checkpoint-final`
- Max length: `1024`
- Seed: `42`
- Prompt set: `llada_smoke`
- GPU: `CUDA_VISIBLE_DEVICES=3`

## Runner

Use the unified script:

```bash
bash examples/benchmarks/sdtt_llada/run_experiment.sh --preset pilot --stage all
```

Supported stages:

- `train`
- `benchmark`
- `all`

Useful flags:

- `--preset smoke|pilot|full`
- `--teacher_model_name_or_path ABS_PATH`
- `--student_model_name_or_path ABS_PATH`
- `--prompt_set llada_smoke`
- `--cuda_device 3`
- `--dry_run`

## Outputs

For preset `<preset>`, the script writes:

- Student checkpoint:
  - `.models/sdtt-llada-<preset>/checkpoint-final`
- Baseline benchmark:
  - `.artifacts/sdtt_llada/<preset>/baseline.json`
- Student benchmark:
  - `.artifacts/sdtt_llada/<preset>/student.json`

The student checkpoint contains `.models/.../sdtt_config.json`,
which allows `sdtt_llada` benchmark runs to recover the stored `student_steps` and
`block_size` automatically.
