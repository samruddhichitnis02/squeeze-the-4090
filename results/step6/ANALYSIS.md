# Online serving load test: bf16 vs fp16 vs fp8 (RTX 4090, vLLM 0.32, Qwen2.5-1.5B-Instruct)

Data: `loadtest_table.md` / `loadtest_table.csv` (full per-rate tables and the summary), `loadtest_curves.png`,
raw per-rate JSON in `raw/`, engine-only control in `engine_only/`, logs in `logs/step6/`.
Method: `vllm serve` per precision, open-loop Poisson arrivals (seed 2122) at 1 to 512 req/s, streamed responses, exactly 128 output
tokens per request, TTFT and TPOT measured from the stream. Rates 112, 144, 160 and 176 come from a second pass (`*_fine` logs) added
to locate the knee. SLA for goodput: TTFT <= 500 ms and TPOT <= 50 ms. Every precision ran one time, so there are no repeats.
FP8 means vLLM online per-tensor FP8 quantization (`--quantization fp8`, CUTLASS FP8 kernel, 112 linear layers quantized, KV cache left in bf16).
The flag is reported as deprecated in favour of `fp8_per_tensor` in this vLLM version.

## Summary

| Precision | Weights (GiB) | KV cache (tokens) | Low-load e2e p50 (2 to 8 req/s) | Low-load TPOT p50 | SLA capacity (req/s, grid-limited) | Peak output tok/s |
|---|---|---|---|---|---|---|
| bf16 | 2.98 | 397,968 | 617 to 671 ms | 4.7 to 5.2 ms | 128 | 18,301 |
| fp16 | 2.98 | 398,112 | 592 to 623 ms | 4.5 to 4.8 ms | 144 | 18,564 |
| fp8 | 1.73 | 436,080 | 497 to 506 ms | 3.8 to 3.9 ms | 144 | 17,265 |

## What the data supports
1. **The online plateau is the serving stack, not the GPU.** All three precisions flatten near 17 to 18.5k output tokens/s
   (about 140 req/s). The same engine without the HTTP/streaming layer reaches 28.7k tokens/s at 256 concurrent sequences
   (`engine_only/`), 57% higher. So at high load the test measures the API server and client path (single-process Python), and the
   SLA capacity of about 128 to 144 req/s is a property of this stack. I did not isolate which side (server or client) is the limit.
2. **bf16 and fp16 are indistinguishable.** They have the same width and memory (2.98 GiB weights, ~398k KV tokens). Their latency differs by
   up to about 16% at some points near the knee (p50 2073 vs 1744 ms at 160 req/s; 1149 vs 1302 ms at 144 req/s), which is a measure of
   run-to-run noise here. Treat single-point differences under roughly 15 to 20% as noise.
3. **FP8 lowers latency at low load.** With a few requests in flight, decode is limited by reading weights from memory, and FP8 halves the
   weight bytes read (weights 1.73 vs 2.98 GiB). Median end-to-end latency is 497 to 506 ms vs 617 to 671 ms (bf16) and 592 to 623 ms (fp16),
   a reduction of roughly 16 to 25%; TPOT is 3.8 to 3.9 ms vs 4.5 to 5.2 ms.
4. **FP8 gives no throughput gain at saturation.** The engine-only run at 256 sequences is identical for bf16 and fp8 (28,754 vs 28,730 tok/s).
   At that batch size this model is no longer limited by weight reads.
5. **FP8 was slower than 16-bit at mid-high load in this test.** At 96 to 128 req/s, FP8 p50 latency is higher than bf16 by about 18 to 21%
   (964 vs 800 ms at 96; 1178 vs 995 ms at 128) and its TPOT is 7.3 to 8.7 ms vs 6.1 to 7.3 ms. The gap is the same size as the largest
   bf16-vs-fp16 difference, so a single point is not convincing, but FP8 is behind at every rate from 96 to 144 req/s (the gap at 144 vs fp16 is
   only 7%). That is suggestive, not established: one run, no repeats, and I did not measure why. Candidate causes (per-step activation
   quantization work, GEMM kernel efficiency at mid batch sizes) are untested.
6. **Memory:** FP8 frees 1.25 GiB of weights, which becomes 9.6% more KV-cache tokens (436k vs 398k). The KV cache itself stays bf16, so the
   larger win is a separate knob (FP8 KV cache) tested in the next step.
7. **Cost per token:** at each precision's SLA capacity the GPU spends about 59 to 65 GPU-seconds per 1M output tokens
   (bf16 64.8, fp16 59.7, fp8 59.4). At an assumed $0.50 per GPU-hour that is about $0.008 to $0.009 per 1M tokens. The price is an illustrative
   assumption (the GPU is owned); GPU-seconds is the price-free number. The three precisions are within noise of each other on cost.

## Context from earlier steps
At low rates the server's median latency (about 610 to 670 ms for bf16) is close to the offline batch-1 figure from the offline benchmark (580 ms),
so the HTTP layer adds roughly 30 to 90 ms. Offline batch 32 reached 6.3k tokens/s; with 256 concurrent sequences the engine reaches 28.7k,
which shows how much of the earlier gap to the online result is concurrency.

## Caveats
- One run per precision; single-point differences smaller than about 15 to 20% are not distinguishable from noise. The low-load FP8 latency
  advantage (16 to 25%, consistent across rates 2 to 16 req/s) is the best supported FP8 effect; the mid-load slowdown is weaker.
- The SLA is loose on TPOT (never above 17 ms observed), so the knee is set by TTFT queueing, not by token speed.
- SLA capacity is limited by the rate grid (1 to 512 req/s with extra points 112, 144, 160, 176). Between points it is unknown.
- The first full run used a coarser grid and the report script failed because matplotlib was missing in the vLLM environment; the raw
  data was intact and the report was regenerated after installing matplotlib there.
- An earlier launch of this step was stopped part-way when it was clear the rate list did not reach saturation. Its results were discarded.
- The model is small (1.5B parameters). The FP8 weight-bandwidth benefit would be larger for bigger models.
