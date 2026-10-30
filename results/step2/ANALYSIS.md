# Serving analysis: HF pipeline vs vLLM

Setup: RTX 4090, Qwen/Qwen2.5-1.5B-Instruct, bf16, greedy decoding, exactly 128 new tokens per request,
40 timed calls per configuration after 3 warmup calls. Full table: `serving_table.md` / `serving_table.csv`.
Latency is per generate call (one request at batch 1, the whole batch at batch 8 and 32).

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|
| HF pipeline, batch 1 | 59.1 | 2158.2 | 2239.5 |
| vLLM, batch 1 | 221.3 | 579.6 | 598.3 |
| vLLM, batch 8 | 1530.7 | 668.1 | 681.2 |
| vLLM, batch 32 | 6272.1 | 652.9 | 655.9 |
| HF pipeline, batch 8 (extra) | 390.4 | 2613.6 | 2699.0 |
| HF pipeline, batch 32 (extra) | 1473.2 | 2775.0 | 2884.6 |

## Question: at which batch size does per-request latency become worse than the baseline while throughput still improves?

Measured against the required baseline (HF pipeline, batch 1, p50 2158 ms), vLLM never gets slower at any batch size
we tested: p50 is 580 ms at batch 1, 668 ms at batch 8 and 653 ms at batch 32, while throughput grows from 221 to
6,272 tokens/sec. The crossover does appear for the HF pipeline itself: at batch 8 its p50 rises to 2,614 ms (21% above its
batch-1 figure) while throughput is 6.6x higher. Within vLLM, going from batch 1 to batch 8 costs 15% more latency
(580 to 668 ms) for 6.9x the throughput, and batch 32 adds nothing further (653 ms, within noise of batch 8).
For an interactive application the per-request latency (p50/p95) is the number that matters, because a user waits
for one response and not for the server's aggregate output; total throughput matters for offline jobs and for cost per token.

## Why the numbers look like this (and limits of this test)

- Decode is memory-bandwidth bound at small batch: each step streams all weights to produce a few tokens, so extra
  sequences in the batch are almost free until compute saturates. That is why latency stays flat while throughput scales.
  A 1.5B model on a 4090 has not reached the compute limit by batch 32, so the latency crossover vLLM would show is not visible yet.
- HF batch-1 speed (59 tokens/sec, about 17 ms per token) is dominated by per-step Python and kernel-launch overhead in eager mode.
  Part of vLLM's 3.7x batch-1 advantage is therefore CUDA graphs and lower overhead, not better math.
- This is an offline test with no request queue. Real online serving adds queueing delay, so p95/p99 under concurrent load
  can behave differently. That is what the later serving study (bonus A) will measure.
- vLLM preallocates 60% of GPU memory for its KV cache, so its peak memory is not comparable to HF's 2.9 to 3.1 GB and was not tabulated.
- The batch 8 versus batch 32 latency difference (668 vs 653 ms) is about 2%, which we treat as noise, not as a real effect.

## vLLM demo observations (own prompts, two models)

Models: Qwen/Qwen2.5-1.5B-Instruct and HuggingFaceTB/SmolLM2-1.7B-Instruct, temperature 0.7, top_p 0.9, seed 2122,
256 max tokens. Four prompts were generated in one batch in about 2.3 to 2.4 s per model. Outputs are in `demo/`.
Both small models made substantive mistakes: SmolLM2 said a 7B model needs about 7 GB (bf16 needs about 14 GB), and both
described "continuous batching" as generic data batching instead of the LLM-serving scheduling technique.
