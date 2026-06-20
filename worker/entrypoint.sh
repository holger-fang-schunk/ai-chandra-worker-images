#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[entrypoint] $*"
}

require_command() {
  local command_name="$1"
  local error_message="$2"

  if ! command -v "${command_name}" >/dev/null 2>&1; then
    log "ERROR: ${error_message}"
    exit 10
  fi

  log "${command_name}: $(command -v "${command_name}")"
}

export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/cache/triton}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64"
fi

export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/workspace/cache/flashinfer}"

mkdir -p \
  /data/in \
  /data/out \
  /data/tmp \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${TRITON_CACHE_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}"

if [[ -z "${AWS_ACCESS_KEY_ID:-}" && -n "${CLOUDFLARE_R2_ACCESS_KEY_ID:-}" ]]; then
  export AWS_ACCESS_KEY_ID="${CLOUDFLARE_R2_ACCESS_KEY_ID}"
fi

if [[ -z "${AWS_SECRET_ACCESS_KEY:-}" && -n "${CLOUDFLARE_R2_SECRET_ACCESS_KEY:-}" ]]; then
  export AWS_SECRET_ACCESS_KEY="${CLOUDFLARE_R2_SECRET_ACCESS_KEY}"
fi

export AWS_REGION="${AWS_REGION:-auto}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8000}"
: "${VLLM_MODEL:=}"
: "${VLLM_MAX_MODEL_LEN:=4096}"
: "${VLLM_MAX_NUM_BATCHED_TOKENS:=1024}"
: "${VLLM_TP:=1}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.80}"
: "${VLLM_STARTUP_TIMEOUT_SECONDS:=900}"
: "${VLLM_EXTRA_ARGS:=}"
: "${VLLM_API_BASE:=http://127.0.0.1:${VLLM_PORT}/v1}"
: "${VLLM_REQUIRE_BUILD_TOOLS:=true}"
: "${VLLM_REQUIRE_NVCC:=true}"

: "${OCR_MODEL_NAME:=${VLLM_MODEL}}"
: "${OCR_PROMPT_TYPE:=ocr_layout}"
: "${OCR_MAX_SIDE:=1400}"
: "${OCR_MAX_NEW_TOKENS:=1500}"
: "${OCR_VLLM_RETRIES:=2}"

MODE="${1:-worker}"
shift || true

case "${MODE}" in
  bash|sh)
    exec "${MODE}" "$@"
    ;;
  sleep|debug)
    log "Debug mode enabled. Container will stay alive."
    exec sleep infinity
    ;;
  worker)
    ;;
  *)
    log "Executing custom command: ${MODE} $*"
    exec "${MODE}" "$@"
    ;;
esac

if [[ -z "${VLLM_MODEL}" ]]; then
  log "ERROR: VLLM_MODEL is not set."
  log "Use mode 'debug' for an interactive container without vLLM."
  exit 2
fi

if [[ "${VLLM_REQUIRE_BUILD_TOOLS}" == "true" ]]; then
  require_command "gcc" "gcc is missing. vLLM/Triton/FlashInfer may need a C compiler."
  require_command "g++" "g++ is missing. vLLM/Triton/FlashInfer may need a C++ compiler."
  require_command "ninja" "ninja is missing. FlashInfer JIT needs ninja-build."
fi

if [[ "${VLLM_REQUIRE_NVCC}" == "true" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    log "ERROR: nvcc is missing. Use a CUDA devel base image, not a runtime base image."
    log "Expected nvcc at: ${CUDA_HOME}/bin/nvcc"
    log "Current CUDA_HOME: ${CUDA_HOME}"
    exit 13
  fi

  log "nvcc: $(command -v nvcc)"
fi

log "Starting vLLM OpenAI-compatible server."
log "Model: ${VLLM_MODEL}"
log "HF_HOME: ${HF_HOME}"
log "HF_HUB_CACHE: ${HF_HUB_CACHE}"
log "XDG_CACHE_HOME: ${XDG_CACHE_HOME}"
log "VLLM_CACHE_ROOT: ${VLLM_CACHE_ROOT}"
log "TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
log "FLASHINFER_WORKSPACE_BASE: ${FLASHINFER_WORKSPACE_BASE}"
log "CUDA_HOME: ${CUDA_HOME}"
log "VLLM_API_BASE: ${VLLM_API_BASE}"

vllm_args=(
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --model "${VLLM_MODEL}"
  --download-dir "${HF_HOME}"
  --tensor-parallel-size "${VLLM_TP}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  # Intentionally split on whitespace for simple extra flags.
  # For complex quoting, prefer baking explicit env vars into this script later.
  # shellcheck disable=SC2206
  extra_args=( ${VLLM_EXTRA_ARGS} )
  vllm_args+=("${extra_args[@]}")
fi

python3 -m vllm.entrypoints.openai.api_server "${vllm_args[@]}" &
vllm_pid=$!

terminate() {
  log "Termination requested. Forwarding signal to vLLM pid ${vllm_pid}."
  kill -TERM "${vllm_pid}" 2>/dev/null || true
  wait "${vllm_pid}" 2>/dev/null || true
  exit 143
}
trap terminate TERM INT

log "Waiting for vLLM readiness. Timeout: ${VLLM_STARTUP_TIMEOUT_SECONDS}s"
start_ts=$(date +%s)

while true; do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    log "vLLM ready."
    break
  fi

  if ! kill -0 "${vllm_pid}" 2>/dev/null; then
    log "ERROR: vLLM process exited before readiness."
    wait "${vllm_pid}" || true
    exit 3
  fi

  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))

  if [[ "${elapsed}" -ge "${VLLM_STARTUP_TIMEOUT_SECONDS}" ]]; then
    log "ERROR: vLLM readiness timeout after ${elapsed}s."
    kill -TERM "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
    exit 4
  fi

  sleep 5
done

worker_args=(
  --model_name "${OCR_MODEL_NAME}"
  --prompt_type "${OCR_PROMPT_TYPE}"
  --max_side "${OCR_MAX_SIDE}"
  --max_new_tokens "${OCR_MAX_NEW_TOKENS}"
  --vllm_api_base "${VLLM_API_BASE}"
  --vllm_retries "${OCR_VLLM_RETRIES}"
)

if [[ "${OCR_NO_HTML:-false}" == "true" ]]; then
  worker_args+=(--no_html)
fi

if [[ "${OCR_NO_METADATA:-false}" == "true" ]]; then
  worker_args+=(--no_metadata)
fi

if [[ "${OCR_LAYOUT_JSON:-true}" == "true" ]]; then
  worker_args+=(--layout_json)
fi

if [[ "${OCR_SAVE_RAW:-false}" == "true" ]]; then
  worker_args+=(--save_raw)
fi

if [[ -n "${OCR_PROMPT_SUFFIX:-}" ]]; then
  worker_args+=(--prompt_suffix "${OCR_PROMPT_SUFFIX}")
fi

if [[ -n "${OCR_SYSTEM_PROMPT:-}" ]]; then
  worker_args+=(--system_prompt "${OCR_SYSTEM_PROMPT}")
fi

if [[ -n "${OCR_S3_BUCKET:-}" ]]; then
  if [[ -z "${OCR_S3_JOB_PREFIX:-}" ]]; then
    log "ERROR: OCR_S3_JOB_PREFIX is required when OCR_S3_BUCKET is set."
    kill -TERM "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
    exit 5
  fi

  worker_args+=(
    --s3_bucket "${OCR_S3_BUCKET}"
    --s3_job_prefix "${OCR_S3_JOB_PREFIX}"
    --s3_download_dir "${OCR_S3_DOWNLOAD_DIR:-/data/in}"
    --s3_work_dir "${OCR_S3_WORK_DIR:-/data/tmp/chandra-ocr-worker}"
  )

  if [[ -n "${OCR_S3_INPUT_PREFIX:-}" ]]; then
    worker_args+=(--s3_input_prefix "${OCR_S3_INPUT_PREFIX}")
  fi

  if [[ -n "${OCR_S3_OUTPUT_PREFIX:-}" ]]; then
    worker_args+=(--s3_output_prefix "${OCR_S3_OUTPUT_PREFIX}")
  fi

  if [[ -n "${OCR_S3_STATE_PREFIX:-}" ]]; then
    worker_args+=(--s3_state_prefix "${OCR_S3_STATE_PREFIX}")
  fi

  if [[ -n "${S3_ENDPOINT_URL:-}" ]]; then
    worker_args+=(--s3_endpoint_url "${S3_ENDPOINT_URL}")
  fi

  if [[ -n "${OCR_MAX_PAGES:-}" ]]; then
    worker_args+=(--limit "${OCR_MAX_PAGES}")
  fi

  if [[ "${OCR_FORCE:-false}" == "true" ]]; then
    worker_args+=(--force)
  fi

  if [[ "${OCR_KEEP_LOCAL:-false}" == "true" ]]; then
    worker_args+=(--s3_keep_local)
  fi
else
  worker_args+=(
    --input_dir "${OCR_INPUT_DIR:-/data/in}"
    --output_dir "${OCR_OUTPUT_DIR:-/data/out}"
  )

  if [[ -n "${OCR_MAX_PAGES:-}" ]]; then
    worker_args+=(--limit "${OCR_MAX_PAGES}")
  fi
fi

log "Starting OCR worker."

set +e
python3 /app/run_chandra_on_images.py "${worker_args[@]}" "$@"
worker_exit=$?
set -e

log "OCR worker exited with code ${worker_exit}. Stopping vLLM."
kill -TERM "${vllm_pid}" 2>/dev/null || true
wait "${vllm_pid}" 2>/dev/null || true
exit "${worker_exit}"
