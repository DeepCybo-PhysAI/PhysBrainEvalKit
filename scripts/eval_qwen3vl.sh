#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_qwen3vl.sh --model-path PATH [options]

Options:
  --model-path PATH         Local Hugging Face model directory.
  --model-name NAME         Name stored in result files.
  --output-base DIR         Unified benchmark output directory.
  --backend hf              Inference backend. Persistent mode currently supports hf only.
  --gpus LIST               GPU IDs, comma or space separated. Default: 0,1,2,3.
  --models-per-gpu N        Resident model processes on each GPU. Default: 4.
  --cpu-per-worker N        CPU cores/threads per worker. Default: auto, capped at 4.
  --only LIST               Run only selected benchmarks.
  --skip LIST               Skip selected benchmarks. Ego3D is always absent from this plan.
  --run-id ID               Persistent shard namespace; reuse it for shard-level resume.
  --prompt-policy POLICY    Input policy: internal or original-when-available. Default: internal.
  --resume                  Skip complete benchmark bundles and reuse valid shards.
  --debug                   Pass --debug to selected benchmark entry points.
  --dry-run                 Validate and print the plan without loading models.
  -h, --help                Show this help.

The persistent plan is Hugging Face-only and excludes Ego3D-Bench, OpenEQA,
and PIO-S3-Verified. With four GPUs and four models per GPU it uses 16 workers.
EOF
}

MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MODELS_PER_GPU="${MODELS_PER_GPU:-4}"
CPU_PER_WORKER="${CPU_PER_WORKER:-}"
ONLY_BENCHMARKS="${ONLY_BENCHMARKS:-}"
SKIP_BENCHMARKS="${SKIP_BENCHMARKS:-Ego3D-Bench}"
EVAL_BACKEND="${EVAL_BACKEND:-hf}"
RUN_ID="${RUN_ID:-}"
PROMPT_POLICY="${PROMPT_POLICY:-internal}"
MMSI_NUM_SAMPLES="${MMSI_NUM_SAMPLES:-1}"
MMSI_SEED="${MMSI_SEED:-3407}"
MMSI_TEMPERATURE="${MMSI_TEMPERATURE:-0.7}"
RESUME=0
DEBUG=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="${2:?--model-path requires a value}"; shift 2 ;;
    --model-path=*) MODEL_PATH="${1#*=}"; shift ;;
    --model-name) MODEL_NAME="${2:?--model-name requires a value}"; shift 2 ;;
    --model-name=*) MODEL_NAME="${1#*=}"; shift ;;
    --output-base) OUTPUT_BASE="${2:?--output-base requires a value}"; shift 2 ;;
    --output-base=*) OUTPUT_BASE="${1#*=}"; shift ;;
    --backend) EVAL_BACKEND="${2:?--backend requires a value}"; shift 2 ;;
    --backend=*) EVAL_BACKEND="${1#*=}"; shift ;;
    --gpus) GPU_IDS="${2:?--gpus requires a value}"; shift 2 ;;
    --gpus=*) GPU_IDS="${1#*=}"; shift ;;
    --models-per-gpu) MODELS_PER_GPU="${2:?--models-per-gpu requires a value}"; shift 2 ;;
    --models-per-gpu=*) MODELS_PER_GPU="${1#*=}"; shift ;;
    --cpu-per-worker) CPU_PER_WORKER="${2:?--cpu-per-worker requires a value}"; shift 2 ;;
    --cpu-per-worker=*) CPU_PER_WORKER="${1#*=}"; shift ;;
    --only) ONLY_BENCHMARKS="${2:?--only requires a value}"; shift 2 ;;
    --only=*) ONLY_BENCHMARKS="${1#*=}"; shift ;;
    --skip) SKIP_BENCHMARKS="${2:?--skip requires a value}"; shift 2 ;;
    --skip=*) SKIP_BENCHMARKS="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --prompt-policy) PROMPT_POLICY="${2:?--prompt-policy requires a value}"; shift 2 ;;
    --prompt-policy=*) PROMPT_POLICY="${1#*=}"; shift ;;
    --mmsi-num-samples) MMSI_NUM_SAMPLES="${2:?--mmsi-num-samples requires a value}"; shift 2 ;;
    --mmsi-num-samples=*) MMSI_NUM_SAMPLES="${1#*=}"; shift ;;
    --mmsi-seed) MMSI_SEED="${2:?--mmsi-seed requires a value}"; shift 2 ;;
    --mmsi-seed=*) MMSI_SEED="${1#*=}"; shift ;;
    --mmsi-temperature) MMSI_TEMPERATURE="${2:?--mmsi-temperature requires a value}"; shift 2 ;;
    --mmsi-temperature=*) MMSI_TEMPERATURE="${1#*=}"; shift ;;
    --resume) RESUME=1; shift ;;
    --debug) DEBUG=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PYTHON="${PYTHON:-python}"
if [[ "$EVAL_BACKEND" != "hf" ]]; then
  echo "ERROR: persistent sharded evaluation currently supports --backend hf only" >&2
  exit 2
fi
if [[ "$PROMPT_POLICY" != "internal" && "$PROMPT_POLICY" != "original" ]]; then
  echo "ERROR: --prompt-policy must be internal or original" >&2
  exit 2
fi
if [[ -z "$MODEL_PATH" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: --model-path must contain config.json: ${MODEL_PATH:-<empty>}" >&2
  exit 2
fi
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
MODEL_NAME="${MODEL_NAME:-$(basename "$(dirname "$MODEL_PATH")")}" 
OUTPUT_BASE="${OUTPUT_BASE:-$(dirname "$MODEL_PATH")/embodiedevalkit}"
OUTPUT_BASE="$("$PYTHON" - "$OUTPUT_BASE" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"

GPU_IDS="${GPU_IDS//,/ }"
read -r -a GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT="${#GPU_ARRAY[@]}"
if (( GPU_COUNT < 1 )); then
  echo "ERROR: at least one GPU is required" >&2
  exit 2
fi
if [[ ! "$MODELS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --models-per-gpu must be a positive integer" >&2
  exit 2
fi
WORLD_SIZE=$((GPU_COUNT * MODELS_PER_GPU))
GPU_LAYOUT=()
for gpu in "${GPU_ARRAY[@]}"; do
  for ((slot = 0; slot < MODELS_PER_GPU; slot++)); do
    GPU_LAYOUT+=("$gpu")
  done
done
GPU_LAYOUT_CSV="$(IFS=,; echo "${GPU_LAYOUT[*]}")"

mapfile -t AVAILABLE_CPU_IDS < <("$PYTHON" - <<'PY'
import os
print(*sorted(os.sched_getaffinity(0)), sep="\n")
PY
)
CPU_BUDGET="${#AVAILABLE_CPU_IDS[@]}"
if [[ -z "$CPU_PER_WORKER" ]]; then
  CPU_PER_WORKER=$((CPU_BUDGET / WORLD_SIZE))
  (( CPU_PER_WORKER > 4 )) && CPU_PER_WORKER=4
fi
if [[ ! "$CPU_PER_WORKER" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --cpu-per-worker must be a positive integer" >&2
  exit 2
fi
if (( WORLD_SIZE * CPU_PER_WORKER > CPU_BUDGET )); then
  echo "ERROR: ${WORLD_SIZE} workers x ${CPU_PER_WORKER} CPUs exceeds the ${CPU_BUDGET}-CPU affinity budget" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  command -v taskset >/dev/null 2>&1 || { echo "ERROR: taskset is unavailable" >&2; exit 1; }
  command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi is unavailable" >&2; exit 1; }
  AVAILABLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  declare -A SEEN_GPUS=()
  for gpu in "${GPU_ARRAY[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ || "$gpu" -ge "$AVAILABLE_GPU_COUNT" ]]; then
      echo "ERROR: GPU ID ${gpu} is outside visible range 0-$((AVAILABLE_GPU_COUNT - 1))" >&2
      exit 2
    fi
    if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
      echo "ERROR: duplicate GPU ID: ${gpu}" >&2
      exit 2
    fi
    SEEN_GPUS[$gpu]=1
  done
fi

RUN_ID="${RUN_ID:-persistent-$(date +%Y%m%d_%H%M%S)}"
ATTEMPT_ID="attempt-$(date +%Y%m%d_%H%M%S)-$$"
RUN_STATE="${OUTPUT_BASE}/.persistent_scheduler/runs/${RUN_ID}"
ATTEMPT_STATE="${RUN_STATE}/attempts/${ATTEMPT_ID}"

COMMON_ARGS=(
  --project-root "$PROJECT_ROOT"
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --output-base "$OUTPUT_BASE"
  --world-size "$WORLD_SIZE"
  --models-per-gpu "$MODELS_PER_GPU"
  --gpu-layout "$GPU_LAYOUT_CSV"
  --cpu-threads "$CPU_PER_WORKER"
  --run-id "$RUN_ID"
  --prompt-policy "$PROMPT_POLICY"
  --mmsi-num-samples "$MMSI_NUM_SAMPLES"
  --mmsi-seed "$MMSI_SEED"
  --mmsi-temperature "$MMSI_TEMPERATURE"
  --attempt-id "$ATTEMPT_ID"
  --skip "$SKIP_BENCHMARKS"
)
[[ -n "$ONLY_BENCHMARKS" ]] && COMMON_ARGS+=(--only "$ONLY_BENCHMARKS")
[[ "$RESUME" == "1" ]] && COMMON_ARGS+=(--resume)
[[ "$DEBUG" == "1" ]] && COMMON_ARGS+=(--debug)

echo "Model: ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Output base: ${OUTPUT_BASE}"
echo "GPU IDs: ${GPU_ARRAY[*]}"
echo "Models per GPU: ${MODELS_PER_GPU}"
echo "Total workers: ${WORLD_SIZE}"
echo "CPU affinity budget: ${CPU_BUDGET}"
echo "CPU per worker: ${CPU_PER_WORKER}"
echo "Reserved worker CPU slots: $((WORLD_SIZE * CPU_PER_WORKER))"
echo "Run ID: ${RUN_ID}"
echo "Prompt policy: ${PROMPT_POLICY}"
echo "Attempt ID: ${ATTEMPT_ID}"

if [[ "$DRY_RUN" == "1" ]]; then
  exec "$PYTHON" "$SCRIPT_DIR/eval_qwen3vl.py" \
    "${COMMON_ARGS[@]}" --worker-id 0 --gpu "${GPU_ARRAY[0]}" --slot 0 --dry-run
fi

mkdir -p "${ATTEMPT_STATE}/startup" "${ATTEMPT_STATE}/failures" "${ATTEMPT_STATE}/logs"
exec 9>"${OUTPUT_BASE}/.persistent_scheduler/launcher.lock"
if ! flock -n 9; then
  echo "ERROR: another persistent evaluation owns ${OUTPUT_BASE}" >&2
  exit 1
fi

declare -a PIDS

terminate_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM
cleanup_on_exit() {
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    terminate_children
  fi
}
trap cleanup_on_exit EXIT

cpu_set_for_worker() {
  local worker_id="$1"
  local first=$((worker_id * CPU_PER_WORKER))
  local ids=()
  local offset
  for ((offset = 0; offset < CPU_PER_WORKER; offset++)); do
    ids+=("${AVAILABLE_CPU_IDS[$((first + offset))]}")
  done
  local joined
  joined="$(IFS=,; echo "${ids[*]}")"
  printf '%s\n' "$joined"
}

start_worker() {
  local worker_id="$1"
  local gpu="$2"
  local slot="$3"
  local cpu_set
  cpu_set="$(cpu_set_for_worker "$worker_id")"
  local log_file="${ATTEMPT_STATE}/logs/worker-$(printf '%02d' "$worker_id").log"
  echo "Starting worker ${worker_id}: GPU ${gpu}, slot ${slot}, CPUs ${cpu_set}"
  taskset -c "$cpu_set" \
    env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE \
      -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
      CUDA_VISIBLE_DEVICES="$gpu" \
      EVAL_CPU_THREADS="$CPU_PER_WORKER" \
      OMP_NUM_THREADS="$CPU_PER_WORKER" \
      MKL_NUM_THREADS="$CPU_PER_WORKER" \
      NUMEXPR_NUM_THREADS="$CPU_PER_WORKER" \
      OPENBLAS_NUM_THREADS=1 \
      BLIS_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 \
      OPENCV_FOR_THREADS_NUM=1 \
      TOKENIZERS_PARALLELISM=false \
      "$PYTHON" "$SCRIPT_DIR/eval_qwen3vl.py" \
        "${COMMON_ARGS[@]}" \
        --worker-id "$worker_id" \
        --gpu "$gpu" \
        --slot "$slot" \
        > "$log_file" 2>&1 &
  PIDS[$worker_id]=$!
}

wait_for_worker() {
  local worker_id="$1"
  local slot="$2"
  local deadline=$((SECONDS + ${STARTUP_TIMEOUT:-7200}))
  local ready="${ATTEMPT_STATE}/startup/slot-${slot}/worker-$(printf '%02d' "$worker_id").ready"
  while [[ ! -f "$ready" ]]; do
    if ! kill -0 "${PIDS[$worker_id]}" 2>/dev/null; then
      echo "ERROR: worker ${worker_id} exited during model startup" >&2
      tail -n 40 "${ATTEMPT_STATE}/logs/worker-$(printf '%02d' "$worker_id").log" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: worker ${worker_id} model startup timed out" >&2
      return 1
    fi
    sleep 5
  done
  echo "Worker ${worker_id}: resident model ready"
}

# Load the same slot on all GPUs in parallel, limiting simultaneous HDFS model
# readers to the number of GPUs instead of the full worker count.
for ((slot = 0; slot < MODELS_PER_GPU; slot++)); do
  for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)); do
    worker_id=$((gpu_index * MODELS_PER_GPU + slot))
    start_worker "$worker_id" "${GPU_ARRAY[$gpu_index]}" "$slot"
  done
  for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)); do
    worker_id=$((gpu_index * MODELS_PER_GPU + slot))
    wait_for_worker "$worker_id" "$slot"
  done
done

printf 'ready %s\n' "$(date -Iseconds)" > "${ATTEMPT_STATE}/startup/all-workers.ready"
echo "All ${WORLD_SIZE} resident models are ready; benchmark sequence begins"

failed=0
for ((worker_id = 0; worker_id < WORLD_SIZE; worker_id++)); do
  wait "${PIDS[$worker_id]}" || failed=$((failed + 1))
done
if (( failed > 0 )); then
  echo "ERROR: ${failed} worker(s) failed; inspect ${ATTEMPT_STATE}" >&2
  exit 1
fi

echo "Persistent sharded evaluation completed"
echo "Output base: ${OUTPUT_BASE}"
echo "Summary: ${OUTPUT_BASE}/summary.txt"
