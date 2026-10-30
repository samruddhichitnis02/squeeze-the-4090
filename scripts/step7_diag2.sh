#!/usr/bin/env bash
# Step 7a (part 2): does per-token streaming limit throughput? Same server (4 API processes), streaming vs non-streaming.
# Run from WSL:  bash scripts/step7_diag2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_vllm.sh

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
LOG=logs/step7
OUT=results/step7/diag
mkdir -p "$LOG" "$OUT"
echo "run started: $(date -Is)" > "$LOG/run_info_diag2.txt"

vllm serve "$MODEL" --port "$PORT" --max-model-len 2048 --gpu-memory-utilization 0.6 --max-num-seqs 256 \
  --seed 2122 --uvicorn-log-level warning --dtype bfloat16 --api-server-count 4 > "$LOG/server_diag2.log" 2>&1 &
pid=$!
up=0
for _ in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
  kill -0 "$pid" 2> /dev/null || break
  sleep 3
done
if [ "$up" -eq 1 ]; then
  for mode in nostream stream; do
    flag=()
    [ "$mode" = "nostream" ] && flag=(--no-stream)
    echo "--- servers=4 clients=2 mode=$mode ---" | tee -a "$LOG/run_info_diag2.txt"
    python -u src/loadtest.py --label "diag2_${mode}" --procs 2 --workload short "${flag[@]}" \
      --rates 256 384 512 --seconds-per-rate 8 --max-requests 4000 --out "$OUT/asc4_${mode}.json" \
      > "$LOG/diag2_${mode}.log" 2>&1 || echo "FAILED mode=$mode" | tee -a "$LOG/run_info_diag2.txt"
    grep '^RATE' "$LOG/diag2_${mode}.log" | python -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l[5:])
    print(f\"  offered {d['offered_rate']:.0f}: achieved {d['achieved_req_per_sec']:.0f} req/s, {d['output_tokens_per_sec']:.0f} tok/s, e2e p50 {d['e2e_ms']['p50']:.0f} ms, failed {d['n_failed']}\")
"
  done
else
  echo "FAILED: server did not come up" | tee -a "$LOG/run_info_diag2.txt"
fi
kill "$pid" 2> /dev/null
wait "$pid" 2> /dev/null
echo "run finished: $(date -Is)" >> "$LOG/run_info_diag2.txt"
