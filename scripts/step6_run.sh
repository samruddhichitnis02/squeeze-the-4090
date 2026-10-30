#!/usr/bin/env bash
# Step 6: online load test of vLLM at bf16, fp16 and fp8. Run from WSL:  bash scripts/step6_run.sh
# Optional refinement pass with a finer rate grid (separate logs/results, merged by the report):
#   RATES="112 144 160 176" SUFFIX=_fine bash scripts/step6_run.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env_vllm.sh

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
LOG=logs/step6
OUT=results/step6
SUF="${SUFFIX:-}"
RATE_ARGS=()
[ -n "${RATES:-}" ] && RATE_ARGS=(--rates $RATES)
mkdir -p "$LOG" "$OUT/raw"

{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
  echo "model: $MODEL"
  echo "rates: ${RATES:-default grid}"
} > "$LOG/run_info${SUF}.txt"
nvidia-smi -L > "$LOG/gpu_uuid${SUF}.txt"
nvidia-smi > "$LOG/nvidia_smi_before${SUF}.txt"
pip freeze > "$LOG/pip_freeze${SUF}.txt"

# label  serve-flags...
serve_and_test() {
  local label=$1
  shift
  echo "=== [$label] vllm serve $* ===" | tee -a "$LOG/run_info${SUF}.txt"
  vllm serve "$MODEL" --port "$PORT" --max-model-len 1024 --gpu-memory-utilization 0.6 \
    --max-num-seqs 256 --seed 2122 --uvicorn-log-level warning "$@" > "$LOG/server_${label}${SUF}.log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
    if ! kill -0 "$pid" 2> /dev/null; then break; fi
    sleep 3
  done
  if [ "$up" -eq 1 ]; then
    echo "[$label] server up, starting load test" | tee -a "$LOG/run_info${SUF}.txt"
    python -u src/loadtest.py --label "$label" --model "$MODEL" --out "$OUT/raw/${label}${SUF}.json" \
      "${RATE_ARGS[@]}" > "$LOG/loadtest_${label}${SUF}.log" 2>&1 \
      || echo "FAILED load test [$label], see $LOG/loadtest_${label}${SUF}.log" | tee -a "$LOG/run_info${SUF}.txt"
    grep '^RATE\|^STOP' "$LOG/loadtest_${label}${SUF}.log" | cut -c1-260
  else
    echo "FAILED: server for [$label] did not come up, see $LOG/server_${label}${SUF}.log" | tee -a "$LOG/run_info${SUF}.txt"
  fi
  kill "$pid" 2> /dev/null
  wait "$pid" 2> /dev/null
  # make sure the GPU is free before the next server starts
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 3000 ] && break
    sleep 2
  done
}

serve_and_test bf16 --dtype bfloat16
serve_and_test fp16 --dtype float16
serve_and_test fp8 --dtype bfloat16 --quantization fp8

nvidia-smi > "$LOG/nvidia_smi_after${SUF}.txt"
python src/make_loadtest_report.py "$OUT" 2>&1 | tee "$LOG/loadtest_report_stdout${SUF}.log"
echo "run finished: $(date -Is)" >> "$LOG/run_info${SUF}.txt"
