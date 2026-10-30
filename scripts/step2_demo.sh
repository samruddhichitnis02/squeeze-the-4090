#!/usr/bin/env bash
# Step 2 (part 2): vLLM inference demo with two different Hugging Face models on your own prompts.
# Run from WSL:  bash scripts/step2_demo.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_vllm.sh
mkdir -p logs/step2 results/step2/demo

for model in "Qwen/Qwen2.5-1.5B-Instruct" "HuggingFaceTB/SmolLM2-1.7B-Instruct"; do
  slug="${model//\//__}"
  echo "=== demo: $model ==="
  python -u src/vllm_demo.py --model "$model" > "logs/step2/demo_${slug}.log" 2>&1 \
    || echo "FAILED: $model (see logs/step2/demo_${slug}.log)"
  grep '^DEMO_DONE' "logs/step2/demo_${slug}.log"
done
