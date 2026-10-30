#!/usr/bin/env bash
# Step 2: serving benchmark (HF pipeline vs vLLM). Run from WSL:  bash scripts/step2_run.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env_vllm.sh

LOG=logs/step2
OUT=results/step2
mkdir -p "$LOG" "$OUT/raw"

{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
} > "$LOG/run_info.txt"
nvidia-smi -L > "$LOG/gpu_uuid.txt"
nvidia-smi > "$LOG/nvidia_smi_before.txt"
pip freeze > "$LOG/pip_freeze.txt"

# name engine batch
for cfg in "hf 1" "vllm 1" "vllm 8" "vllm 32" "hf 8" "hf 32"; do
  set -- $cfg
  echo "=== running engine=$1 batch=$2 ===" | tee -a "$LOG/run_info.txt"
  python -u src/serve_bench.py --engine "$1" --batch-size "$2" --out "$OUT/raw" \
    > "$LOG/bench_$1_bs$2.log" 2>&1 || echo "FAILED: $1 bs$2 (see $LOG/bench_$1_bs$2.log)" | tee -a "$LOG/run_info.txt"
  grep '^RESULT' "$LOG/bench_$1_bs$2.log" | tail -1
done

nvidia-smi > "$LOG/nvidia_smi_after.txt"
python src/make_serving_table.py "$OUT" | tee "$LOG/serving_table_stdout.log"
echo "run finished: $(date -Is)" >> "$LOG/run_info.txt"
