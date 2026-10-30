#!/usr/bin/env bash
# Quick check that vLLM starts and generates. Writes to /tmp only.
cd "$(dirname "$0")/.."
source scripts/env_vllm.sh
python -u src/serve_bench.py --engine vllm --batch-size 8 --num-batches 2 --warmup 1 --out /tmp/smoke > /tmp/smoke.log 2>&1
grep -E '^RESULT|RuntimeError|fatal error|No such file' /tmp/smoke.log | tail -6 | cut -c1-500
