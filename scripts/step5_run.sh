#!/usr/bin/env bash
# Step 5: full fine-tune (all 45,615 train-split rows, 70/15/15 split, 3 epochs) under four configurations,
# to measure real GPU-hours and confirm accuracy. Run from WSL:  bash scripts/step5_run.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$HOME/gpu-assignment/.venv/bin/activate"

LOG=logs/step5
OUT=results/step5
mkdir -p "$LOG" "$OUT"

{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
} > "$LOG/run_info.txt"
nvidia-smi -L > "$LOG/gpu_uuid.txt"
nvidia-smi > "$LOG/nvidia_smi_before.txt"
pip freeze > "$LOG/pip_freeze.txt"

# name  extra-flags...
run_cfg() {
  local name=$1
  shift
  mkdir -p "$OUT/$name"
  echo "=== [$name] full fine-tune: $* ===" | tee -a "$LOG/run_info.txt"
  python -u src/train_bert.py --n_rows 45615 --epochs 3 --skip-save --out "$OUT/$name" "$@" \
    > "$LOG/$name.log" 2>&1 || echo "FAILED [$name], see $LOG/$name.log" | tee -a "$LOG/run_info.txt"
}

run_cfg fp32_bs32 --batch_size 32
run_cfg bf16_bs32 --batch_size 32 --bf16
run_cfg bf16_bs128 --batch_size 128 --bf16
run_cfg fp32_bs128 --batch_size 128

nvidia-smi > "$LOG/nvidia_smi_after.txt"
echo "run finished: $(date -Is)" >> "$LOG/run_info.txt"
