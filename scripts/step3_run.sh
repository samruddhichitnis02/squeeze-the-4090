#!/usr/bin/env bash
# Step 3: batch-size scaling to the memory wall. Run from WSL:  bash scripts/step3_run.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$HOME/gpu-assignment/.venv/bin/activate"

LOG=logs/step3
OUT=results/step3
mkdir -p "$LOG" "$OUT/raw"
rm -f "$OUT"/raw/*.json   # start clean so stale sizes never mix into the table

{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
} > "$LOG/run_info.txt"
nvidia-smi -L > "$LOG/gpu_uuid.txt"
nvidia-smi > "$LOG/nvidia_smi_before.txt"
pip freeze > "$LOG/pip_freeze.txt"

MAX_BS=65536
bs=8
while [ "$bs" -le "$MAX_BS" ]; do
  echo "=== batch size $bs ===" | tee -a "$LOG/run_info.txt"
  timeout 300 python -u src/batch_scale.py --batch-size "$bs" --out "$OUT/raw" > "$LOG/bs${bs}.log" 2>&1
  code=$?
  if [ "$code" -eq 124 ]; then
    echo "TIMEOUT (300 s) at batch size $bs: likely GPU memory spilling to system RAM" | tee -a "$LOG/run_info.txt"
    break
  fi
  grep '^RESULT' "$LOG/bs${bs}.log" | tail -1 | cut -c1-400
  if [ "$code" -eq 3 ]; then
    echo "out of memory at batch size $bs: stopping sweep" | tee -a "$LOG/run_info.txt"
    break
  elif [ "$code" -ne 0 ]; then
    echo "FAILED (exit $code) at batch size $bs: see $LOG/bs${bs}.log" | tee -a "$LOG/run_info.txt"
    break
  fi
  bs=$((bs * 2))
done

nvidia-smi > "$LOG/nvidia_smi_after.txt"
python src/make_scaling_report.py "$OUT" | tee "$LOG/scaling_report_stdout.log"
echo "run finished: $(date -Is)" >> "$LOG/run_info.txt"
