# Step 7: serving knobs and a roofline check (RTX 4090, vLLM 0.32, Qwen2.5-1.5B-Instruct)

Data: `knob_report.md` (summary and full per-rate tables, roofline tables), `knob_results.csv`, `knob_curves.png`,
`diag/diag_table.md` (bottleneck diagnostic), raw runs in `raw/` and `diag/`, logs in `logs/step7/`.
Method: one `vllm serve` per configuration (bf16 unless stated, 4 API server processes, 2 client processes, max-model-len 2048,
128 output tokens, streamed, open-loop Poisson arrivals), two workloads: **short** (about 41 prompt tokens, decode dominated) and
**longprefix** (about 951 prompt tokens, a ~900-token document shared by all requests plus a unique question).
One run per cell, so single differences under roughly 15 to 20% are noise (see Step 6). Hardware ceilings: 1,008 GB/s memory bandwidth and
165.6 TFLOP/s bf16 tensor throughput (dense, FP32 accumulate; TechPowerUp measurement of a stock 4090).

## 1. What limited the Step 6 plateau
| Setup | tok/s at 384 offered req/s |
|---|---|
| 1 API server, 1 client (Step 6 setup) | 18,689 |
| 1 API server, 4 client processes | 15,786 |
| 4 API servers, 1 client | 21,472 |
| 4 API servers, 4 clients | 20,349 |
| 4 API servers, 2 clients, non-streaming | 21,366 |
| engine only, no HTTP (Step 6 control, 256 sequences) | 28,754 |

More API server processes raised the plateau by about 15% (18.7k to 21.5k). More client processes and disabling streaming changed
nothing (or made it worse). The remaining gap to the engine-only 28.7k (about 25%) is not explained: I did not profile the API
path. Consequence: absolute online capacities below depend on this stack; comparisons between configurations on the same stack are still valid.

## 2. Findings by knob
**Concurrency cap (`--max-num-seqs`): the clearest effect.** On the short workload, peak throughput was 19,166 tok/s at 256, 10,534 at 64 and
5,987 at 32. Decode throughput is about (sequences in flight) / (time per step). Predicted from the measured TPOT at 96 req/s:
64 / 6.0 ms = 10.7k and 32 / 5.2 ms = 6.2k tok/s, within 3% of the measured peaks. On the long-prefix workload a request lasts about
0.03 s + 127 x 6.0 ms = 0.79 s, so a cap of 32 predicts 32 / 0.79 = 40.5 req/s, against 38.8 measured (peak 4,962 tok/s, SLA capacity 32 versus at
least 64 req/s at 256 seqs). A cap of 64 matched the 256 result on long-prefix (7,236 vs 7,262 tok/s) but hit TTFT p99 of 1,211 ms at 64 req/s
(versus 56 ms). A cap that is too low silently limits capacity and shows up as queueing in TTFT, not as slower tokens.

**Prefix caching: very large on the shared-prefix workload, small on short prompts.** Long-prefix with caching off: SLA capacity 16 req/s
versus at least 64 (the ladder top, so the true gap is larger), peak 3,110 versus 7,262 tok/s (-57%), TTFT p99 at 32 req/s 3,141 ms versus 29 ms.
Below saturation (8 to 16 req/s) the two are close (e2e p50 820 vs 704 ms at 8 req/s). With caching off, saturated prefill ran at
about 70 TFLOP/s, 42% of the bf16 tensor peak (24.3 req/s x (951 x 2.62 + 128 x 3.09) GFLOP). On the short workload caching off cost 8% of peak
throughput (17,553 vs 19,166) and raised TTFT p99 at 160 req/s from 1,093 to 2,145 ms; that is suggestive only (one run, near the noise
level). It is plausible because every prompt starts with the same chat-template system prompt.

**Chunked prefill and `--max-num-batched-tokens`: no effect when prefix caching hides the prefill, a latency trade-off when it does not.**
With caching on, both knobs matched the baseline on both workloads (there is little prefill to chunk). The fair test turns caching off.
Below saturation (8 and 16 req/s) all three cache-off variants are the same within noise. Under overload (32 req/s offered, 24 achieved),
a 512-token budget gave TPOT p50 18.7 ms versus 48.1 ms for the default, but TTFT p50 1,262 ms versus 211 ms: smaller prefill chunks let running decodes
proceed between chunks (lower inter-token time) while new prompts wait longer for their first token. Peak throughput was unchanged (3,210 vs 3,110 tok/s).
Disabling chunked prefill gave no clear difference from the default. These comparisons are in the overload region, where queueing amplifies everything,
so treat the size of the effect as approximate and the direction as the finding.

**KV cache precision (FP8 KV, Triton attention backend): doubles capacity, no measurable speed change here.** FP8 KV raised KV capacity from
386,048 to 772,400 tokens (2.0x; consistent with 28,672 bytes per token for this model: 28 layers x 2 (K and V) x 2 KV heads x 128 dims x 2 bytes, 11.1 GB
either way). Peak throughput and latency matched the same-backend bf16-KV control: 22,131 vs 21,950 tok/s, long-prefix TPOT p50 5.3 vs 5.3 ms. This is a smaller effect than a
bandwidth model predicts, and I have no confirmed explanation. At about 45 sequences of about 1,100 tokens the KV reads are about 1.4 GB per step in bf16 (45 x 1,100 x 28,672 B),
so FP8 KV would remove about 0.7 GB, which is 0.7 ms at 1,008 GB/s, up to 13% of a 5.3 ms step. Observed: no change. Possible reasons (untested): fewer sequences in flight than
assumed, a large fixed per-step cost (Section 3 estimates about 2 ms), or the Triton attention kernel converting FP8 to bf16 in a way that does not save time. The measured benefit is
capacity (2.0x KV tokens), which these workloads did not need.
**Accuracy of FP8 KV was not measured.** vLLM warns that it can lose accuracy without calibrated scaling factors and none were supplied; this goes into the Step 8 accuracy tests.

**FP8 weights: lower latency at low batch, no gain (and weaker results) at high load.** Consistent with Step 6. Under load, FP8 weights peaked at 17,322 tok/s vs 19,166
(-10%, FlashAttention backend) and 18,685 vs 21,950 (-15%, Triton backend), and TTFT p99 at 160 req/s was worse. FP8 weights also freed memory: KV capacity 425,136 tokens
versus 397,968 (+6.8%). On the long-prefix workload the TPOT gain was inconsistent (4.7 vs 6.0 ms on FlashAttention, 5.2 vs 5.3 ms on Triton), so I do not
claim one.

**Attention backend: a confound I did not intend.** FP8 KV cache is not supported by the default FlashAttention backend, and the fallback (FlashInfer) failed to build in this
environment, so those runs use the Triton backend. The Triton control looked better than the default backend at high load (peak 21,950 vs 19,166 tok/s, +15%; SLA capacity 160 vs 96;
TTFT p99 at 160 req/s 55 vs 1,093 ms). This is one run per backend and queueing near the knee magnifies small differences, so I do not conclude Triton is faster. It does mean
FP8-KV numbers must be compared with the Triton controls, which is how they are reported above.

## 3. Roofline cross-checks
- **Decode at about 1 request in flight is memory-bandwidth bound, with a fixed overhead.** bf16 reads about 3.09 GB of weights per step (1.31B layer params plus the 0.23B
  tied lm_head, 2 bytes each): 3.06 ms at 1,008 GB/s. Observed TPOT (Step 6, 2 req/s) is 4.74 ms, 65% of the bandwidth bound. FP8 weights read 1.78 GB (the 112 transformer linears
  at 1 byte, lm_head stays bf16): bound 1.76 ms, observed 3.82 ms, 46%. A pure bandwidth model predicts FP8 at 58% of bf16's step time; observed is 81%.
  A fixed cost of about 1.7 to 2.1 ms per step (kernel launches, attention, sampling, norms) that does not shrink with weight bytes would explain this. That is an inference from two points, not a profile.
- **Throughput scales with batch while decode is bandwidth bound**, then flattens against the compute and stack limits: the sequence-cap results above follow tokens/s = seqs / TPOT closely.
- **Compute ceiling:** engine-only at 256 sequences did 28.7k tok/s, which is 89 TFLOP/s counting only decode matmuls (54% of 165.6) and up to about 113 TFLOP/s (68%) if prompt
  tokens are included without crediting prefix caching. The online plateau (18 to 22k tok/s) is 56 to 68 TFLOP/s decode-only (34 to 41% of peak), so the online stack, not the GPU, set it.
  Saturated prefill with caching off reached 42 to 44% of peak, below the engine-only decode-heavy figure; I did not profile why.
- **KV cache sizing:** 386,048 tokens x 28,672 bytes = 11.1 GB, consistent with the 0.6 memory budget minus weights and activation/graph memory. FP8 gives exactly 2.0x tokens for the same bytes.

## 4. Caveats
- One run per cell. Differences under about 15 to 20% are noise, and many comparisons near the overload knee are noisier than that.
- The rate grids are coarse (short: 32, 96, 160, 256; long-prefix: 8, 16, 32, 64). Configurations marked ">= ladder top" were not pushed to failure.
- Cache-off and cap results at the highest rates are overload measurements dominated by queueing.
- The FP8-KV runs use a different attention backend; accuracy was not measured for FP8 weights or FP8 KV (Step 8).
- The two FlashAttention-backend FP8-KV launches (`kv_fp8`, `weights_fp8_kv_fp8`) failed to start (FlashInfer build error: nvcc 13.4 vs CUDA headers 13.0); their server logs are kept in `logs/step7/`.
- Absolute online capacities are limited by the Python API stack (see section 1).
