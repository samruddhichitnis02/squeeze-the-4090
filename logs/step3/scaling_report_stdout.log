# Batch scaling to the memory wall

GPU: NVIDIA GeForce RTX 4090 (24.0 GB total). BERT fine-tune, fp32, max_len 64, AdamW lr 2e-05. 20 timed steps per batch size after 3 warmup steps, one fresh process per size.
Memory wall definition: PyTorch allocations are capped at 21.3 GB (95% of the 22.4 GB free at start; Windows holds the rest for the display). The cap makes overflow a real out-of-memory error, since the Windows driver would otherwise spill to system RAM.
Batches larger than the 3,500 train rows are sampled with replacement. Epoch time is estimated as 3,500 / samples-per-second (not a measured epoch).

| Batch size | Status | Samples/sec | Peak memory allocated (GB) | Peak reserved (GB) | Est. epoch time (s) | Gain vs previous size |
|---|---|---|---|---|---|---|
| 8 | ok | 306.9 | 2.06 | 2.27 | 11.41 |  |
| 16 | ok | 490.8 | 2.09 | 2.26 | 7.13 | +59.9% |
| 32 | ok | 678.9 | 2.47 | 2.82 | 5.16 | +38.3% |
| 64 | ok | 876.5 | 3.71 | 4.00 | 3.99 | +29.1% |
| 128 | ok | 980.1 | 6.09 | 6.27 | 3.57 | +11.8% |
| 256 | ok | 1024.5 | 10.94 | 11.14 | 3.42 | +4.5% |
| 512 | ok | 1081.4 | 20.50 | 20.67 | 3.24 | +5.6% |
| 1024 | **OOM** | n/a | n/a | n/a | n/a | n/a |

- Largest power-of-two batch size that runs: **512** (first failure at 1024)
- Highest throughput: 1081.4 samples/sec at batch 512
- Throughput stops improving (gain under 5% per doubling) at batch size **256**
- Linear fit of peak memory vs batch size: about 1.45 GB fixed (weights, gradients, optimizer states, CUDA context overhead) + 38.0 MB per sample (max abs residual 0.32 GB)
- Fit predicts the wall near batch 534 for the 21.3 GB usable cap
