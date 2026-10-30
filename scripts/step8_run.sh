#!/usr/bin/env bash
# Step 8: quality + speed for each precision / quantisation configuration, one vLLM process per configuration.
# Run from WSL:  bash scripts/step8_run.sh        (ONLY="label1 label2" runs a subset; SUFFIX tags the run_info file)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env_vllm.sh

LOG=logs/step8
OUT=results/step8/raw
SUF="${SUFFIX:-}"
mkdir -p "$LOG" "$OUT"
{
  echo "run started: $(date -Is)"
  echo "host: $(hostname)"
  echo "python: $(python --version 2>&1)"
} > "$LOG/run_info${SUF}.txt"
nvidia-smi -L > "$LOG/gpu_uuid${SUF}.txt"
nvidia-smi > "$LOG/nvidia_smi_before${SUF}.txt"
pip freeze > "$LOG/pip_freeze${SUF}.txt"

# label | flags for quant_eval.py
CONFIGS=(
  "bf16_ref|"
  "bf16_ref2|"
  "fp16|--dtype float16"
  "fp8_weights|--quantization fp8"
  "triton_bf16|--attention-backend TRITON_ATTN"
  "triton_fp8kv|--attention-backend TRITON_ATTN --kv-cache-dtype fp8"
  "triton_fp8kv_e5m2|--attention-backend TRITON_ATTN --kv-cache-dtype fp8_e5m2"
  "triton_fp8kv_pth|--attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head"
  "triton_w8_fp8kv|--attention-backend TRITON_ATTN --quantization fp8 --kv-cache-dtype fp8"
  "awq_int4|--model Qwen/Qwen2.5-1.5B-Instruct-AWQ --dtype float16"
  "gptq_int4|--model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 --dtype float16"
)

for cfg in "${CONFIGS[@]}"; do
  label=${cfg%%|*}
  flags=${cfg#*|}
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $label "* ]]; then continue; fi
  echo "=== [$label] quant_eval $flags ===" | tee -a "$LOG/run_info${SUF}.txt"
  # shellcheck disable=SC2086
  python -u src/quant_eval.py --label "$label" --out "$OUT" $flags > "$LOG/eval_${label}.log" 2>&1 \
    || echo "FAILED [$label], see $LOG/eval_${label}.log" | tee -a "$LOG/run_info${SUF}.txt"
  grep -E '^(SPEED|PPL|MMLU|GSM8K)' "$LOG/eval_${label}.log" | cut -c1-260
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 3000 ] && break
    sleep 2
  done
done

nvidia-smi > "$LOG/nvidia_smi_after${SUF}.txt"
echo "run finished: $(date -Is)" >> "$LOG/run_info${SUF}.txt"
