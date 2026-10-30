#!/usr/bin/env bash
# Step 1 (HW4.a): fine-tune BERT, capturing environment, GPU UUID and full logs.
# Run from WSL:  bash scripts/step1_run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$HOME/gpu-assignment/.venv/bin/activate"

OUT=results/step1
LOG=logs/step1
mkdir -p "$OUT" "$LOG"

{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
  echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo 'no commit yet')"
} > "$LOG/run_info.txt"

nvidia-smi -L > "$LOG/gpu_uuid.txt"
nvidia-smi > "$LOG/nvidia_smi_before.txt"
pip freeze > "$LOG/pip_freeze.txt"

python -u src/train_bert.py --out "$OUT" 2>&1 | tee "$LOG/train_stdout.log"

nvidia-smi > "$LOG/nvidia_smi_after.txt"
python src/summarize_results.py "$OUT"
echo "run finished: $(date -Is)" >> "$LOG/run_info.txt"
