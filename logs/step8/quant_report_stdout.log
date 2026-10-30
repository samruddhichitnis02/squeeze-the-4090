# Step 8: quantisation, speed and measured quality (RTX 4090, Qwen2.5-1.5B-Instruct, vLLM 0.32)

Quality sets: WikiText-2 test perplexity (298,646 tokens, window 1024), MMLU zero-shot all subjects (n=14,042), GSM8K zero-shot chain-of-thought (n=1,319, greedy, 512 tokens max).
Differences are **paired against the bf16 reference question by question**; the standard error (SE) comes from the per-question differences. "Within noise" means the paired difference is under 2 SE. The row "bf16 repeat" runs the identical configuration a second time and shows what pure run-to-run variation looks like.

## Quality

| Configuration | Perplexity | MMLU (95% CI) | MMLU diff vs bf16 (paired) | GSM8K (95% CI) | GSM8K diff vs bf16 (paired) | Questions better / worse (GSM8K) | GSM8K outputs identical to bf16 |
|---|---|---|---|---|---|---|---|
| bf16 (reference) | 10.709 | 58.45% (±0.82) | reference | 74.45% (±2.35) | reference | n/a | n/a |
| bf16 repeat (noise floor) | 10.709 | 58.45% (±0.82) | -0.01 pts, within noise | 74.22% (±2.36) | -0.23 pts, within noise | 38 / 41 | 46.4% |
| fp16 | 10.700 | 58.29% (±0.82) | -0.16 pts, within noise | 75.13% (±2.33) | +0.68 pts, within noise | 44 / 35 | 40.3% |
| FP8 weights | 10.813 | 58.24% (±0.82) | -0.21 pts, within noise | 75.13% (±2.33) | +0.68 pts, within noise | 84 / 75 | 3.6% |
| [Triton] bf16 (control) | 10.709 | 58.40% (±0.82) | -0.05 pts, within noise | 73.92% (±2.37) | -0.53 pts, within noise | 36 / 43 | 42.8% |
| [Triton] FP8 KV, e4m3 fixed scale | 30.421 | 52.44% (±0.83) | -6.01 pts, worse (17.5 SE) | 3.11% (±0.94) | -71.34 pts, worse (55.7 SE) | 8 / 949 | 0.0% |
| [Triton] FP8 KV, per-token-head scale | 49.781 | 41.11% (±0.81) | -17.34 pts, worse (36.5 SE) | 1.52% (±0.66) | -72.93 pts, worse (58.7 SE) | 4 / 966 | 0.0% |
| [Triton] FP8 weights + FP8 KV (e4m3) | 30.962 | 51.96% (±0.83) | -6.49 pts, worse (18.0 SE) | 4.32% (±1.10) | -70.13 pts, worse (53.0 SE) | 14 / 939 | 0.0% |
| INT4 AWQ (weight-only) | 11.620 | 55.36% (±0.82) | -3.09 pts, worse (10.2 SE) | 69.14% (±2.49) | -5.31 pts, worse (4.7 SE) | 78 / 148 | 0.1% |
| INT4 GPTQ (weight-only) | 11.973 | 55.93% (±0.82) | -2.53 pts, worse (8.2 SE) | 67.78% (±2.52) | -6.67 pts, worse (5.5 SE) | 85 / 173 | 0.1% |

## Speed and memory (same processes)

Batch-1 time per output token is the median of 12 single-request calls of 128 tokens. Batch-256 throughput is the median of 4 engine-only runs (no HTTP layer) with 256 concurrent sequences. Weights are what vLLM reports as loaded; KV capacity is tokens that fit in the 0.6 memory budget.

| Configuration | Attention backend | Weights (GiB) | KV cache (tokens) | Batch-1 ms/token | vs bf16 | Batch-256 tokens/s | vs bf16 |
|---|---|---|---|---|---|---|---|
| bf16 (reference) | default | 2.98 | 384,208 | 4.57 | 1.00x | 28,392 | 1.00x |
| bf16 repeat (noise floor) | default | 2.98 | 384,208 | 4.59 | 1.00x | 28,314 | 1.00x |
| fp16 | default | 2.98 | 383,200 | 4.60 | 1.01x | 26,530 | 0.93x |
| FP8 weights | default | 1.73 | 422,912 | 3.91 | 0.86x | 28,094 | 0.99x |
| [Triton] bf16 (control) | TRITON_ATTN | 2.98 | 383,056 | 4.24 | 0.93x | 29,884 | 1.05x |
| [Triton] FP8 KV, e4m3 fixed scale | TRITON_ATTN | 2.98 | 768,736 | 4.25 | 0.93x | 31,087 | 1.09x |
| [Triton] FP8 KV, per-token-head scale | TRITON_ATTN | 2.98 | 742,896 | 4.31 | 0.94x | 31,348 | 1.10x |
| [Triton] FP8 weights + FP8 KV (e4m3) | TRITON_ATTN | 1.73 | 845,248 | 3.68 | 0.81x | 29,958 | 1.06x |
| INT4 AWQ (weight-only) | default | 1.1 | 454,944 | 2.53 | 0.55x | 30,014 | 1.06x |
| INT4 GPTQ (weight-only) | default | 1.1 | 453,856 | 2.54 | 0.55x | 29,376 | 1.03x |

## Run-to-run noise floor (bf16 vs bf16, identical settings)

GSM8K accuracy difference -0.23 pts (SE 0.67); MMLU difference -0.01 pts (SE 0.02); GSM8K outputs identical for 46.4% of problems; batch-1 ms/token 4.59 vs 4.57; batch-256 tokens/s 28,314 vs 28,392. Any configuration difference smaller than this spread is not evidence of an effect.
