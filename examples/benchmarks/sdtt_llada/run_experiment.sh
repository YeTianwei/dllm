#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_PATH="/home/timer/miniconda3/envs/dllm"
DEFAULT_CHECKPOINT="${ROOT}/.models/smoke_test_llada_sft/checkpoint-final"
DEFAULT_ACCELERATE_CONFIG="${ROOT}/scripts/accelerate_configs/zero3.yaml"

PRESET="pilot"
STAGE="all"
PROMPT_SET="llada_smoke"
TEACHER_MODEL_NAME_OR_PATH="${DEFAULT_CHECKPOINT}"
STUDENT_MODEL_NAME_OR_PATH="${DEFAULT_CHECKPOINT}"
CUDA_DEVICE="3"
DRY_RUN="false"
NUM_GPUS="1"
ACCELERATE_CONFIG="${DEFAULT_ACCELERATE_CONFIG}"
OUTPUT_TAG=""

resolve_repo_path() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
  else
    printf '%s\n' "${ROOT}/${input_path}"
  fi
}

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
    --num_gpus)
      NUM_GPUS="$2"; shift 2 ;;
    --accelerate_config)
      ACCELERATE_CONFIG="$2"; shift 2 ;;
    --train_batch_size)
      OVERRIDE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --grad_acc_steps)
      OVERRIDE_GRAD_ACC_STEPS="$2"; shift 2 ;;
    --max_length)
      OVERRIDE_MAX_LENGTH="$2"; shift 2 ;;
    --teacher_steps)
      OVERRIDE_TEACHER_STEPS="$2"; shift 2 ;;
    --student_steps)
      OVERRIDE_STUDENT_STEPS="$2"; shift 2 ;;
    --block_size)
      OVERRIDE_BLOCK_SIZE="$2"; shift 2 ;;
    --output_tag)
      OUTPUT_TAG="$2"; shift 2 ;;
    --dtype)
      OVERRIDE_DTYPE="$2"; shift 2 ;;
    --load_in_4bit)
      OVERRIDE_LOAD_IN_4BIT="$2"; shift 2 ;;
    --gradient_checkpointing)
      OVERRIDE_GRADIENT_CHECKPOINTING="$2"; shift 2 ;;
    --dry_run)
      DRY_RUN="true"; shift 1 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

TEACHER_MODEL_NAME_OR_PATH="$(resolve_repo_path "${TEACHER_MODEL_NAME_OR_PATH}")"
STUDENT_MODEL_NAME_OR_PATH="$(resolve_repo_path "${STUDENT_MODEL_NAME_OR_PATH}")"
ACCELERATE_CONFIG="$(resolve_repo_path "${ACCELERATE_CONFIG}")"

if [[ -n "${OVERRIDE_DTYPE:-}" ]]; then
  OVERRIDE_DTYPE_ARG="--dtype ${OVERRIDE_DTYPE}"
else
  OVERRIDE_DTYPE_ARG=""
fi
if [[ -n "${OVERRIDE_LOAD_IN_4BIT:-}" ]]; then
  OVERRIDE_LOAD_IN_4BIT_ARG="--load_in_4bit ${OVERRIDE_LOAD_IN_4BIT}"
else
  OVERRIDE_LOAD_IN_4BIT_ARG=""
fi
if [[ -n "${OVERRIDE_GRADIENT_CHECKPOINTING:-}" ]]; then
  OVERRIDE_GRADIENT_CHECKPOINTING_ARG="--gradient_checkpointing ${OVERRIDE_GRADIENT_CHECKPOINTING}"
else
  OVERRIDE_GRADIENT_CHECKPOINTING_ARG=""
fi

case "${PRESET}" in
  smoke)
    DATASET_ARGS='tatsu-lab/alpaca[train:8,test:4]'
    MAX_STEPS=1
    TEACHER_STEPS=16
    STUDENT_STEPS=4
    BLOCK_SIZE=4
    MAX_LENGTH=512
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

# Apply value overrides after preset selection so they are not overwritten.
if [[ -n "${OVERRIDE_TRAIN_BATCH_SIZE:-}" ]]; then
  TRAIN_BATCH_SIZE="${OVERRIDE_TRAIN_BATCH_SIZE}"
fi
if [[ -n "${OVERRIDE_GRAD_ACC_STEPS:-}" ]]; then
  GRAD_ACC_STEPS="${OVERRIDE_GRAD_ACC_STEPS}"
fi
if [[ -n "${OVERRIDE_MAX_LENGTH:-}" ]]; then
  MAX_LENGTH="${OVERRIDE_MAX_LENGTH}"
fi
if [[ -n "${OVERRIDE_TEACHER_STEPS:-}" ]]; then
  TEACHER_STEPS="${OVERRIDE_TEACHER_STEPS}"
fi
if [[ -n "${OVERRIDE_STUDENT_STEPS:-}" ]]; then
  STUDENT_STEPS="${OVERRIDE_STUDENT_STEPS}"
fi
if [[ -n "${OVERRIDE_BLOCK_SIZE:-}" ]]; then
  BLOCK_SIZE="${OVERRIDE_BLOCK_SIZE}"
fi

OUTPUT_SUFFIX="${PRESET}"
if [[ -n "${OUTPUT_TAG}" ]]; then
  OUTPUT_SUFFIX="${PRESET}-${OUTPUT_TAG}"
fi

OUTPUT_DIR="${ROOT}/.models/sdtt-llada-${OUTPUT_SUFFIX}"
ARTIFACT_DIR="${ROOT}/.artifacts/sdtt_llada/${OUTPUT_SUFFIX}"
BASELINE_JSON="${ARTIFACT_DIR}/baseline.json"
STUDENT_JSON="${ARTIFACT_DIR}/student.json"
TRAIN_SCRIPT="${ROOT}/examples/benchmarks/train_sdtt_llada.py"
BENCH_SCRIPT="${ROOT}/examples/benchmarks/run_llada_benchmark.py"

# Calculate number of visible GPUs / processes
COUNT_GPUS=$(echo "${CUDA_DEVICE}" | tr ',' '\n' | wc -l)
if [[ "${NUM_GPUS}" != "1" ]]; then
  COUNT_GPUS="${NUM_GPUS}"
fi

if [[ "${COUNT_GPUS}" -gt 1 ]]; then
  TRAIN_LAUNCHER="accelerate launch --config_file ${ACCELERATE_CONFIG} --num_processes ${COUNT_GPUS}"
else
  TRAIN_LAUNCHER="python"
fi

TRAIN_CMD=$(cat <<EOF
cd ${ROOT}
source ~/.zshrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
${TRAIN_LAUNCHER} ${TRAIN_SCRIPT} \
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
  --report_to none \
  ${OVERRIDE_DTYPE_ARG} \
  ${OVERRIDE_LOAD_IN_4BIT_ARG} \
  ${OVERRIDE_GRADIENT_CHECKPOINTING_ARG}
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
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --steps 24 \
  --block_size 24
EOF
)

echo "Preset: ${PRESET}"
echo "Stage: ${STAGE}"
echo "Teacher checkpoint: ${TEACHER_MODEL_NAME_OR_PATH}"
echo "Student init checkpoint: ${STUDENT_MODEL_NAME_OR_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Artifacts dir: ${ARTIFACT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICE}"
echo "Number of GPUs: ${COUNT_GPUS}"
echo "Train launcher: ${TRAIN_LAUNCHER}"

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
