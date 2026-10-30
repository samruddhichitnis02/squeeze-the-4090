#!/usr/bin/env bash
# Step 7b: serving knob sweep. One knob changed per server configuration, each tested on two workloads:
#   short       short chat prompts (decode dominated)
#   longprefix  ~1000-token shared document + unique question (prefill / KV-read heavy)
# Common server settings: bf16, 4 API server processes, max-model-len 2048, gpu-memory-utilization 0.6.
# Run from WSL:  bash scripts/step7_run.sh      (set ONLY="label1 label2" to run a subset)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env_vllm.sh

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
LOG=logs/step7
OUT=results/step7/raw
mkdir -p "$LOG" "$OUT"
SUF="${SUFFIX:-}"

{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
  echo "model: $MODEL"
} > "$LOG/run_info${SUF}.txt"
nvidia-smi -L > "$LOG/gpu_uuid${SUF}.txt"
nvidia-smi > "$LOG/nvidia_smi_before${SUF}.txt"
pip freeze > "$LOG/pip_freeze${SUF}.txt"

SHORT_RATES="32 96 160 256"
LONG_RATES="8 16 32 64"

# label  serve-flags...
run_cfg() {
  local label=$1
  shift
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $label "* ]]; then return; fi
  echo "=== [$label] vllm serve $* ===" | tee -a "$LOG/run_info${SUF}.txt"
  vllm serve "$MODEL" --port "$PORT" --max-model-len 2048 --gpu-memory-utilization 0.6 --seed 2122 \
    --uvicorn-log-level warning --dtype bfloat16 --api-server-count 4 "$@" > "$LOG/server_${label}${SUF}.log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
    kill -0 "$pid" 2> /dev/null || break
    sleep 3
  done
  if [ "$up" -eq 1 ]; then
    for wl in ${WORKLOADS:-short longprefix}; do
      local rates=$SHORT_RATES
      [ "$wl" = "longprefix" ] && rates=$LONG_RATES
      echo "[$label] load test workload=$wl rates=$rates" | tee -a "$LOG/run_info${SUF}.txt"
      python -u src/loadtest.py --label "$label" --workload "$wl" --procs 2 --rates $rates \
        --seconds-per-rate 15 --min-requests 80 --max-requests 2400 --out "$OUT/${label}__${wl}${SUF}.json" \
        > "$LOG/loadtest_${label}__${wl}${SUF}.log" 2>&1 \
        || echo "FAILED load test [$label/$wl], see $LOG/loadtest_${label}__${wl}${SUF}.log" | tee -a "$LOG/run_info${SUF}.txt"
      grep '^STOP' "$LOG/loadtest_${label}__${wl}${SUF}.log" | tee -a "$LOG/run_info${SUF}.txt"
    done
  else
    echo "FAILED: server for [$label] did not come up, see $LOG/server_${label}${SUF}.log" | tee -a "$LOG/run_info${SUF}.txt"
  fi
  kill "$pid" 2> /dev/null
  wait "$pid" 2> /dev/null
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 3000 ] && break
    sleep 2
  done
}

run_cfg base --max-num-seqs 256
run_cfg no_prefix_cache --max-num-seqs 256 --no-enable-prefix-caching
run_cfg no_chunked_prefill --max-num-seqs 256 --no-enable-chunked-prefill
run_cfg batched_tokens_512 --max-num-seqs 256 --max-num-batched-tokens 512
run_cfg seqs_32 --max-num-seqs 32
run_cfg seqs_64 --max-num-seqs 64
run_cfg kv_fp8 --max-num-seqs 256 --kv-cache-dtype fp8
run_cfg weights_fp8 --max-num-seqs 256 --quantization fp8
# FP8 KV cache is not supported by the default FlashAttention backend and the FlashInfer fallback failed to build here
# (nvcc/CUDA header mismatch), so the FP8-KV runs use the Triton attention backend. Controls with the SAME backend and a
# bf16 KV cache separate the backend effect from the KV-precision effect.
run_cfg triton_kv_bf16 --max-num-seqs 256 --attention-backend TRITON_ATTN
run_cfg triton_kv_fp8 --max-num-seqs 256 --attention-backend TRITON_ATTN --kv-cache-dtype fp8
run_cfg triton_w8_kv_bf16 --max-num-seqs 256 --attention-backend TRITON_ATTN --quantization fp8
run_cfg triton_w8_kv_fp8 --max-num-seqs 256 --attention-backend TRITON_ATTN --quantization fp8 --kv-cache-dtype fp8
# With prefix caching on, the long-prefix workload does almost no prefill, so chunked prefill and the batched-token limit have
# nothing to act on. These runs turn prefix caching off so prefill is real, then vary those two knobs (long-prefix workload only):
#   ONLY='nocache_nochunk nocache_bt512' WORKLOADS=longprefix SUFFIX=_nc bash scripts/step7_run.sh
run_cfg nocache_nochunk --max-num-seqs 256 --no-enable-prefix-caching --no-enable-chunked-prefill
run_cfg nocache_bt512 --max-num-seqs 256 --no-enable-prefix-caching --max-num-batched-tokens 512

nvidia-smi > "$LOG/nvidia_smi_after${SUF}.txt"
echo "run finished: $(date -Is)" >> "$LOG/run_info${SUF}.txt"
