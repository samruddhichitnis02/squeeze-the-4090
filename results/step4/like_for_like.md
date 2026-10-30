# Like-for-like comparison at fixed batch sizes

Each cell: samples/sec | peak memory GB (OOM = out of memory, - = not run).

| Technique | batch 64 | batch 256 | batch 512 | batch 1024 | batch 2048 |
|---|---|---|---|---|---|
| baseline | 876 \| 3.71 | 1046 \| 10.94 | 1106 \| 20.50 | OOM | - |
| bf16 | 1506 \| 3.03 | 2235 \| 7.64 | 2307 \| 13.86 | OOM | - |
| grad_ckpt | 688 \| 2.08 | 764 \| 2.84 | 804 \| 4.40 | 815 \| 7.56 | 823 \| 13.86 |
| grad_accum | 891 \| 3.71 | 992 \| 4.14 | 1013 \| 4.14 | 1024 \| 4.14 | 1028 \| 4.14 |
| adam8bit | 853 \| 3.07 | 1029 \| 10.32 | 1081 \| 19.90 | OOM | - |
| combo_bf16_adam8bit | 1376 \| 2.42 | 2177 \| 7.03 | 2279 \| 13.24 | OOM | - |
| combo_bf16_ckpt_adam8bit | 1189 \| 1.18 | 1739 \| 1.93 | 1793 \| 3.22 | 1841 \| 5.79 | 1855 \| 10.93 |

## Ratios vs baseline at batch 512 (where both ran)

- bf16: throughput 2.09x, memory 0.68x (-6.64 GB)
- grad_ckpt: throughput 0.73x, memory 0.21x (-16.10 GB)
- grad_accum: throughput 0.92x, memory 0.20x (-16.36 GB)
- adam8bit: throughput 0.98x, memory 0.97x (-0.61 GB)
- combo_bf16_adam8bit: throughput 2.06x, memory 0.65x (-7.26 GB)
- combo_bf16_ckpt_adam8bit: throughput 1.62x, memory 0.16x (-17.28 GB)
