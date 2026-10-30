# Serving comparison

Model: Qwen/Qwen2.5-1.5B-Instruct, bf16, greedy, 128 new tokens per request (EOS suppressed), 40 timed calls per configuration after warmup.

Latency is per generate call. At batch 1 that is one request; at batch 8/32 it is the whole batch.

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) | Requests/sec | Throughput vs HF b1 | p50 latency vs HF b1 |
|---|---|---|---|---|---|---|
| HF pipeline, batch 1 | 59.1 | 2158.2 | 2239.5 | 0.46 | 1.00x | 1.00x |
| vLLM, batch 1 | 221.3 | 579.6 | 598.3 | 1.73 | 3.74x | 0.27x |
| vLLM, batch 8 | 1530.7 | 668.1 | 681.2 | 11.96 | 25.90x | 0.31x |
| vLLM, batch 32 | 6272.1 | 652.9 | 655.9 | 49.00 | 106.13x | 0.30x |
| HF pipeline, batch 8 (extra, not required) | 390.4 | 2613.6 | 2699.0 | 3.05 | 6.61x | 1.21x |
| HF pipeline, batch 32 (extra, not required) | 1473.2 | 2775.0 | 2884.6 | 11.51 | 24.93x | 1.29x |
