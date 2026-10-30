#!/usr/bin/env bash
# Quick check of the riskiest configurations with tiny eval sizes. Writes to /tmp only.
# Run from WSL:  bash scripts/step8_smoke.sh
cd "$(dirname "$0")/.."
source scripts/env_vllm.sh
for cfg in "triton_fp8kv|--attention-backend TRITON_ATTN --kv-cache-dtype fp8" "awq|--model Qwen/Qwen2.5-1.5B-Instruct-AWQ --dtype float16"; do
  label=${cfg%%|*}
  flags=${cfg#*|}
  echo "=== smoke $label ==="
  python -u src/quant_eval.py --label "smoke_$label" --limit-mmlu 200 --limit-gsm 40 --ppl-tokens 20000 --out /tmp/smoke8 $flags \
    > "/tmp/smoke8_$label.log" 2>&1
  grep -E '^(SPEED|PPL|MMLU|GSM8K|EVAL_DONE)|Error|error:' "/tmp/smoke8_$label.log" | grep -v 'core.py' | tail -8 | cut -c1-300
done
