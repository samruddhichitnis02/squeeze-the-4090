#!/usr/bin/env bash
# Step 4: memory techniques one at a time. For each technique:
#   (1) batch-size sweep to the memory wall (fresh process per size)
#   (2) real fine-tune at the HW4 setting (batch 32, 4 epochs) for accuracy, peak memory and wall-clock
# Run from WSL:  bash scripts/step4_run.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$HOME/gpu-assignment/.venv/bin/activate"

LOG=logs/step4
OUT=results/step4
mkdir -p "$LOG" "$OUT"

{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
} > "$LOG/run_info.txt"
nvidia-smi -L > "$LOG/gpu_uuid.txt"
nvidia-smi > "$LOG/nvidia_smi_before.txt"
pip freeze > "$LOG/pip_freeze.txt"

MAX_BS=16384

# sweep NAME START_BS CAP_BS MICRO_BATCH(or 0) FLAGS...
sweep() {
  local name=$1 bs=$2 cap=$3 micro=$4
  shift 4
  local raw="$OUT/$name/raw" lg="$LOG/$name"
  mkdir -p "$raw" "$lg"
  rm -f "$raw"/*.json
  local microflag=()
  [ "$micro" -gt 0 ] && microflag=(--micro-batch "$micro")
  while [ "$bs" -le "$cap" ]; do
    local steps=20
    [ "$bs" -ge 1024 ] && steps=6
    echo "=== [$name] sweep batch $bs ===" | tee -a "$LOG/run_info.txt"
    timeout 600 python -u src/batch_scale.py --batch-size "$bs" --timed-steps "$steps" --out "$raw" \
      "${microflag[@]}" "$@" > "$lg/bs${bs}.log" 2>&1
    local code=$?
    if [ "$code" -eq 124 ]; then
      echo "TIMEOUT at [$name] batch $bs (possible memory spill)" | tee -a "$LOG/run_info.txt"; break
    elif [ "$code" -eq 3 ]; then
      echo "[$name] out of memory at batch $bs" | tee -a "$LOG/run_info.txt"; break
    elif [ "$code" -ne 0 ]; then
      echo "FAILED (exit $code) [$name] batch $bs: see $lg/bs${bs}.log" | tee -a "$LOG/run_info.txt"; break
    fi
    bs=$((bs * 2))
  done
}

# quality NAME FLAGS...   (HW4 setting: batch 32, 4 epochs, seed 2122)
quality() {
  local name=$1
  shift
  mkdir -p "$OUT/$name/quality" "$LOG/$name"
  echo "=== [$name] accuracy run ===" | tee -a "$LOG/run_info.txt"
  python -u src/train_bert.py --out "$OUT/$name/quality" --skip-save "$@" > "$LOG/$name/quality.log" 2>&1 \
    || echo "FAILED accuracy run [$name], see $LOG/$name/quality.log" | tee -a "$LOG/run_info.txt"
}

sweep baseline 8 $MAX_BS 0
quality baseline

sweep bf16 8 $MAX_BS 0 --bf16
quality bf16 --bf16

sweep grad_ckpt 8 $MAX_BS 0 --grad-ckpt
quality grad_ckpt --grad-ckpt

# accumulation: fixed micro-batch 64, effective batch grows without a memory limit; cap at 2048
sweep grad_accum 64 2048 64
quality grad_accum --accum-steps 4

sweep adam8bit 8 $MAX_BS 0 --optim adamw8bit
quality adam8bit --optim adamw8bit

sweep combo_bf16_adam8bit 8 $MAX_BS 0 --bf16 --optim adamw8bit
quality combo_bf16_adam8bit --bf16 --optim adamw8bit

sweep combo_bf16_ckpt_adam8bit 8 $MAX_BS 0 --bf16 --grad-ckpt --optim adamw8bit
quality combo_bf16_ckpt_adam8bit --bf16 --grad-ckpt --optim adamw8bit

nvidia-smi > "$LOG/nvidia_smi_after.txt"
python src/make_technique_report.py "$OUT" | tee "$LOG/technique_report_stdout.log"
echo "run finished: $(date -Is)" >> "$LOG/run_info.txt"
