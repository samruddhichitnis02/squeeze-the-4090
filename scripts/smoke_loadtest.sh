#!/usr/bin/env bash
# Quick check of the vLLM server + load-test client. Writes to /tmp only.
cd "$(dirname "$0")/.."
source scripts/env_vllm.sh
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --max-model-len 1024 --gpu-memory-utilization 0.6 \
  --seed 2122 --uvicorn-log-level warning --dtype bfloat16 > /tmp/smoke_server.log 2>&1 &
pid=$!
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && break
  kill -0 $pid 2> /dev/null || break
  sleep 3
done
python -u src/loadtest.py --label smoke --rates 4 --min-requests 20 --warmup-requests 8 --out /tmp/smoke_lt.json 2>&1 | tail -5 | cut -c1-500
kill $pid 2> /dev/null
wait $pid 2> /dev/null
echo SMOKE_DONE
