# GPU-hours for one full fine-tune (RTX 4090, 1 GPU)

Full fine-tune = all 45,615 train-split rows, split 70/15/15 with seed 2122: 31930 train / 6842 val / 6843 test, 3 epochs.

## How it is calculated

```
samples processed      = train rows x epochs
                       = 31930 x 3 = 95790
predicted seconds      = samples processed / step throughput (samples/sec, from the Step 4 sweep at that batch size)
GPU-hours              = wall-clock seconds / 3600   (1 GPU, so GPU-hours = hours)
```

Worked example, fp32, batch 32 (HW4 baseline): 95790 / 669.8 = 143 s predicted. Measured: 144 s of pure training plus validation after each epoch = 152 s wall-clock = 152 / 3600 = 0.0421 GPU-hours.

| Configuration | Step throughput (samples/s) | Predicted (s) | Measured training only (s) | Measured wall-clock incl. validation (s) | GPU-hours | Peak memory (GB) | Best epoch | Test accuracy |
|---|---|---|---|---|---|---|---|---|
| fp32, batch 32 (HW4 baseline) | 670 | 143 | 144 | 152 | 0.0421 | 2.47 | 1 | 72.82% |
| bf16, batch 32 | 1090 | 88 | 98 | 101 | 0.0279 | 2.25 | 1 | 72.78% |
| bf16, batch 128 | 2087 | 46 | 48 | 51 | 0.0142 | 4.55 | 3 | 73.18% |
| fp32, batch 128 | 972 | 99 | 95 | 102 | 0.0284 | 6.09 | 3 | 73.32% |

Wall-clock is what the GPU was reserved for, so GPU-hours use it. The prediction covers training steps only, so the gap between predicted and measured-training-only shows how well step throughput from the sweep carries over to a real run (data transfer from the CPU, optimizer on the last partial batch, and the DataLoader are included in the real run).
