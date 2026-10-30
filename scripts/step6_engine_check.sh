#!/usr/bin/env bash
# Step 6 control: vLLM engine throughput with 256 concurrent sequences and NO HTTP/streaming layer.
# If this is well above the server's ~18k tok/s plateau, the plateau is the API server / client, not the GPU.
# Run from WSL:  bash scripts/step6_engine_check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_vllm.sh
LOG=logs/step6
OUT=results/step6/engine_only
mkdir -p "$LOG" "$OUT"
nvidia-smi -L > "$LOG/gpu_uuid_engine_check.txt"

echo "run started: $(date -Is)" > "$LOG/run_info_engine_check.txt"
for cfg in "bf16:" "fp8:fp8"; do
  tag=${cfg%%:*}
  q=${cfg#*:}
  qflag=()
  [ -n "$q" ] && qflag=(--quantization "$q")
  echo "=== engine-only batch 256 [$tag] ===" | tee -a "$LOG/run_info_engine_check.txt"
  python -u src/serve_bench.py --engine vllm --batch-size 256 --num-batches 10 --warmup 3 \
    --out "$OUT" --tag "_$tag" "${qflag[@]}" > "$LOG/engine_check_${tag}.log" 2>&1 \
    || echo "FAILED [$tag], see $LOG/engine_check_${tag}.log" | tee -a "$LOG/run_info_engine_check.txt"
  grep '^RESULT' "$LOG/engine_check_${tag}.log" | tail -1 | cut -c1-420
done
echo "run finished: $(date -Is)" >> "$LOG/run_info_engine_check.txt"
