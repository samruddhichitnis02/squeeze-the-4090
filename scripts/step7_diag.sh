#!/usr/bin/env bash
# Step 7a: find what limited the Step 6 plateau (~18k tok/s) by crossing API-server processes x client processes.
# Run from WSL:  bash scripts/step7_diag.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_vllm.sh

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
LOG=logs/step7
OUT=results/step7/diag
mkdir -p "$LOG" "$OUT"
{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
} > "$LOG/run_info_diag.txt"
nvidia-smi -L > "$LOG/gpu_uuid.txt"
pip freeze > "$LOG/pip_freeze.txt"

for asc in 1 4; do
  echo "=== api-server-count=$asc ===" | tee -a "$LOG/run_info_diag.txt"
  vllm serve "$MODEL" --port "$PORT" --max-model-len 2048 --gpu-memory-utilization 0.6 --max-num-seqs 256 \
    --seed 2122 --uvicorn-log-level warning --dtype bfloat16 --api-server-count "$asc" \
    > "$LOG/server_diag_asc${asc}.log" 2>&1 &
  pid=$!
  up=0
  for _ in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
    kill -0 "$pid" 2> /dev/null || break
    sleep 3
  done
  if [ "$up" -eq 1 ]; then
    for procs in 1 4; do
      echo "--- servers=$asc clients=$procs ---" | tee -a "$LOG/run_info_diag.txt"
      python -u src/loadtest.py --label "diag_asc${asc}_procs${procs}" --procs "$procs" --workload short \
        --rates 192 256 384 --seconds-per-rate 8 --max-requests 3200 --out "$OUT/asc${asc}_procs${procs}.json" \
        > "$LOG/diag_asc${asc}_procs${procs}.log" 2>&1 \
        || echo "FAILED asc=$asc procs=$procs, see $LOG/diag_asc${asc}_procs${procs}.log" | tee -a "$LOG/run_info_diag.txt"
      grep '^RATE' "$LOG/diag_asc${asc}_procs${procs}.log" | python -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l[5:])
    print(f\"  offered {d['offered_rate']:.0f}: achieved {d['achieved_req_per_sec']:.0f} req/s, {d['output_tokens_per_sec']:.0f} tok/s, e2e p50 {d['e2e_ms']['p50']:.0f} ms, failed {d['n_failed']}\")
"
    done
  else
    echo "FAILED: server asc=$asc did not come up, see $LOG/server_diag_asc${asc}.log" | tee -a "$LOG/run_info_diag.txt"
  fi
  kill "$pid" 2> /dev/null
  wait "$pid" 2> /dev/null
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 3000 ] && break
    sleep 2
  done
done
echo "run finished: $(date -Is)" >> "$LOG/run_info_diag.txt"
