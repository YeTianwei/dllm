#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ytw/VLA_baseline/dllm"
ENV_PATH="/home/timer/miniconda3/envs/dllm"
DEFAULT_CHECKPOINT="${ROOT}/.models/smoke_test_llada_sft/checkpoint-final"

PRESET="pilot"
STAGE="all"
PROMPT_SET="llada_smoke"
TEACHER_MODEL_NAME_OR_PATH="${DEFAULT_CHECKPOINT}"
STUDENT_MODEL_NAME_OR_PATH="${DEFAULT_CHECKPOINT}"
CUDA_DEVICE="3"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="$2"; shift 2 ;;
    --stage)
      STAGE="$2"; shift 2 ;;
    --prompt_set)
      PROMPT_SET="$2"; shift 2 ;;
    --teacher_model_name_or_path)
      TEACHER_MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --student_model_name_or_path)
      STUDENT_MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --cuda_device)
      CUDA_DEVICE="$2"; shift 2 ;;
    --dry_run)
      DRY_RUN="true"; shift 1 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

case "${PRESET}" in
  smoke)
    DATASET_ARGS='tatsu-lab/alpaca[train:8,test:4]'
    MAX_STEPS=1
    TEACHER_STEPS=16
    STUDENT_STEPS=4
    BLOCK_SIZE=4
    MAX_LENGTH=1024
    LEARNING_RATE=2e-5
    TRAIN_BATCH_SIZE=1
    EVAL_BATCH_SIZE=1
    GRAD_ACC_STEPS=1
    LOGGING_STEPS=1
    SAVE_STEPS=10
    SEED=42
    MAX_NEW_TOKENS=8
    NUM_REPEATS=1
    WARMUP_RUNS=0
    ;;
  pilot)
    DATASET_ARGS='allenai/tulu-3-sft-mixture[train:1024,test:128]'
    MAX_STEPS=200
    TEACHER_STEPS=64
    STUDENT_STEPS=8
    BLOCK_SIZE=8
    MAX_LENGTH=1024
    LEARNING_RATE=2e-5
    TRAIN_BATCH_SIZE=1
    EVAL_BATCH_SIZE=1
    GRAD_ACC_STEPS=1
    LOGGING_STEPS=1
    SAVE_STEPS=100
    SEED=42
    MAX_NEW_TOKENS=64
    NUM_REPEATS=3
    WARMUP_RUNS=1
    ;;
  full)
    DATASET_ARGS='allenai/tulu-3-sft-mixture[train:4096,test:256]'
    MAX_STEPS=1000
    TEACHER_STEPS=64
    STUDENT_STEPS=8
    BLOCK_SIZE=8
    MAX_LENGTH=1024
    LEARNING_RATE=2e-5
    TRAIN_BATCH_SIZE=1
    EVAL_BATCH_SIZE=1
    GRAD_ACC_STEPS=1
    LOGGING_STEPS=10
    SAVE_STEPS=200
    SEED=42
    MAX_NEW_TOKENS=64
    NUM_REPEATS=3
    WARMUP_RUNS=1
    ;;
  *)
    echo "Unsupported preset: ${PRESET}" >&2
    exit 1 ;;
esac

OUTPUT_DIR="${ROOT}/.models/sdtt-llada-${PRESET}"
ARTIFACT_DIR="${ROOT}/.artifacts/sdtt_llada/${PRESET}"
BASELINE_JSON="${ARTIFACT_DIR}/baseline.json"
STUDENT_JSON="${ARTIFACT_DIR}/student.json"
TRAIN_SCRIPT="${ROOT}/examples/benchmarks/train_sdtt_llada.py"
BENCH_SCRIPT="${ROOT}/examples/benchmarks/run_llada_benchmark.py"

TRAIN_CMD=$(cat <<EOF
cd ${ROOT}
source ~/.zshrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
python ${TRAIN_SCRIPT} \
  --teacher_model_name_or_path ${TEACHER_MODEL_NAME_OR_PATH} \
  --student_model_name_or_path ${STUDENT_MODEL_NAME_OR_PATH} \
  --dataset_args "${DATASET_ARGS}" \
  --max_length ${MAX_LENGTH} \
  --max_steps ${MAX_STEPS} \
  --teacher_steps ${TEACHER_STEPS} \
  --student_steps ${STUDENT_STEPS} \
  --block_size ${BLOCK_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
  --logging_steps ${LOGGING_STEPS} \
  --seed ${SEED} \
  --output_dir ${OUTPUT_DIR} \
  --save_steps ${SAVE_STEPS} \
  --eval_strategy no \
  --report_to none
EOF
)

BASELINE_CMD=$(cat <<EOF
cd ${ROOT}
source ~/.zshrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
python ${BENCH_SCRIPT} \
  --method baseline_llada \
  --model_name_or_path ${TEACHER_MODEL_NAME_OR_PATH} \
  --prompt_set ${PROMPT_SET} \
  --output_json ${BASELINE_JSON} \
  --num_repeats ${NUM_REPEATS} \
  --warmup_runs ${WARMUP_RUNS} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --steps ${TEACHER_STEPS} \
  --block_size ${TEACHER_STEPS}
EOF
)

STUDENT_CMD=$(cat <<EOF
cd ${ROOT}
source ~/.zshrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
python ${BENCH_SCRIPT} \
  --method sdtt_llada \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-final \
  --prompt_set ${PROMPT_SET} \
  --output_json ${STUDENT_JSON} \
  --num_repeats ${NUM_REPEATS} \
  --warmup_runs ${WARMUP_RUNS} \
  --max_new_tokens ${MAX_NEW_TOKENS}
EOF
)

echo "Preset: ${PRESET}"
echo "Stage: ${STAGE}"
echo "Teacher checkpoint: ${TEACHER_MODEL_NAME_OR_PATH}"
echo "Student init checkpoint: ${STUDENT_MODEL_NAME_OR_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Artifacts dir: ${ARTIFACT_DIR}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo
  echo "[TRAIN]"
  echo "${TRAIN_CMD}"
  echo
  echo "[BASELINE BENCHMARK]"
  echo "${BASELINE_CMD}"
  echo
  echo "[STUDENT BENCHMARK]"
  echo "${STUDENT_CMD}"
  exit 0
fi

mkdir -p "${ARTIFACT_DIR}"

run_cmd() {
  local cmd="$1"
  bash -lc "${cmd}"
}

case "${STAGE}" in
  train)
    run_cmd "${TRAIN_CMD}" ;;
  benchmark)
    run_cmd "${BASELINE_CMD}"
    run_cmd "${STUDENT_CMD}" ;;
  all)
    run_cmd "${TRAIN_CMD}"
    run_cmd "${BASELINE_CMD}"
    run_cmd "${STUDENT_CMD}" ;;
  *)
    echo "Unsupported stage: ${STAGE}" >&2
    exit 1 ;;
esac
