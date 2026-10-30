# Summary: training throughput and memory engineering

Hardware: one NVIDIA GeForce RTX 4090 (24 GB, about 22.4 GB free after the Windows display), GPU UUID
`GPU-c5ad1a63-b7c0-1838-5d8f-0df577ec82ce`. Model: `bert-base-uncased` fine-tuned on `cardiffnlp/tweet_eval` (sentiment), seed 2122.
These numbers are HW4.5 measurements on this card. 

Only one card was used, so the second card, the cross-card throughput comparison and the max-batch versus VRAM ratio were not measured.

## Best configurations

| Goal | Best configuration | Measured value | Where |
|---|---|---|---|
| Lowest peak memory | bf16 + gradient checkpointing + 8-bit optimiser (Combination B) | 1.08 GB at batch 32; 3.22 GB at batch 512 (baseline: 2.47 GB and 20.50 GB) | `results/step4/technique_table.md`, `like_for_like.md` |
| Highest throughput | bf16 autocast | 2306.6 samples/s at batch 512 (baseline 1105.9) | `results/step4/technique_table.md` |
| Shortest wall-clock, HW4 fine-tune (3,500 rows, 4 epochs, batch 32) | bf16 autocast | 15.6 s (baseline 22.4 s) | `results/step4/technique_table.md` |
| Shortest wall-clock, full fine-tune (31,930 rows, 3 epochs) | bf16 autocast, batch 128 | 51 s = 0.0142 GPU-hours (baseline 152 s = 0.0421) | `results/step5/gpu_hours.md` |
| Largest batch | gradient checkpointing, or Combination B | 2048 (baseline 512) | `results/step4/technique_table.md` |

## Table  (RTX 4090)

Throughput and peak memory are measured at each technique's own max batch. Accuracy is test accuracy after a real fine-tune at the HW4 setting.

| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |
|---|---|---|---|---|
| Baseline (HW4 settings) | 512 | 1105.9 | 20.50 | 66.53% |
| Mixed precision (bf16) | 512 | 2306.6 | 13.86 | 66.53% |
| Gradient checkpointing | 2048 | 823.3 | 13.86 | 66.53% |
| Gradient accumulation (micro-batch 64) | 2048+ (sweep cap, no OOM) | 1027.6 | 4.14 | 66.67% |
| 8-bit optimiser | 512 | 1081.3 | 19.90 | 67.20% |
| Best combination (A: bf16 + 8-bit optimiser) | 512 | 2278.9 | 13.24 | 65.87% |
| Best combination (B: bf16 + checkpointing + 8-bit optimiser) | 2048 | 1854.6 | 10.93 | 65.87% |

The accuracy differences are within noise (750 test rows, one seed, standard error about 1.7 points), so no technique is shown to change accuracy.
Details: `results/step4/ANALYSIS.md`.

## Batch scaling (Part A, baseline settings)
Largest power-of-two batch that runs: 512 (20.50 GB, 1081 samples/s). Batch 1024 fails with a CUDA out-of-memory error.
Throughput stops improving at batch 256 (under 5% gain per doubling); the curve is flat from about 128. Peak memory is about
1.45 GB fixed plus 38 MB per sample. Details: `results/step3/ANALYSIS.md`.

## GPU-hours for one full fine-tune
Full fine-tune = all 45,615 train-split rows split 70/15/15 (31,930 train rows), 3 epochs, 1 GPU.

```
samples processed = 31,930 x 3 = 95,790
predicted seconds = 95,790 / step throughput        (baseline: 95,790 / 669.8 = 143 s)
GPU-hours         = measured wall-clock seconds / 3600   (baseline: 152 / 3600 = 0.0421)
```

| Configuration | Predicted (s) | Measured wall-clock (s) | GPU-hours | Peak memory (GB) | Test accuracy |
|---|---|---|---|---|---|
| fp32, batch 32 (HW4 baseline) | 143 | 152 | 0.0421 | 2.47 | 72.82% |
| bf16, batch 32 | 88 | 101 | 0.0279 | 2.25 | 72.78% |
| **bf16, batch 128** | 46 | 51 | **0.0142** | 4.55 | 73.18% |
| fp32, batch 128 | 99 | 102 | 0.0284 | 6.09 | 73.32% |

The prediction from the Step 4 sweep is within about 10% of the measured training time (1% for the baseline, the largest gap is bf16 at batch 32: 88 s versus 98 s). Wall-clock includes validation after each epoch.

## Recommended configuration for a full fine-tune on this hardware
**bf16 autocast, batch size 128, fp32 AdamW, no checkpointing, accumulation or 8-bit optimiser, 3 epochs (keep the best-validation checkpoint).**

Justification from the tables:
- bf16 is the only technique in Table HW4.5.1 that raises throughput (2.09x at batch 512), and it cuts memory by a third. Checkpointing,
  accumulation and the 8-bit optimiser all lower or leave throughput unchanged, and their benefit (memory) is not needed here.
- Memory is not the constraint on this card: bf16 at batch 128 uses 4.55 GB of about 22 GB, so there is no reason to pay
  checkpointing's 27% slowdown.
- Batch 128 is at the throughput knee (bf16 reaches 2087 samples/s there, 90% of the 2307 at batch 512) at a fraction of the memory.
- The full-data runs confirm it: 3.0x fewer GPU-hours than the baseline (0.0142 versus 0.0421) with test accuracy 73.18% versus 72.82%, a difference within noise.

Limits of this recommendation: one seed, learning rate not tuned for batch 128 (2e-5 was kept), and validation loss was lowest at epoch 1 in all four
full runs, so fewer epochs would save more time if the best checkpoint is all that is needed.


