# Memory techniques analysis (single card: RTX 4090)

Data: `technique_table.md` (Table HW4.5.1 plus the extended table), `like_for_like.md`, `technique_comparison.png`,
per-technique sweeps in `<technique>/raw/`, accuracy runs in `<technique>/quality/`, logs in `logs/step4/`.
Each technique was switched on alone against the baseline (Step 1 settings: BERT-base, fp32, AdamW, max_len 64),
then two combinations were tried. Sweep setup is the same as Step 3 (fresh process per batch size, allocator capped at
21.3 GB usable so overflow is a real out-of-memory error). Accuracy comes from a separate real fine-tune per technique
at the HW4 setting (batch 32, 4 epochs, seed 2122).

## What each technique did (batch 512, where baseline still runs)

| Technique | Throughput vs baseline | Peak memory vs baseline | Max batch (baseline 512) |
|---|---|---|---|
| Mixed precision (bf16 autocast) | 2.09x | 0.68x (-6.6 GB) | 512 |
| Gradient checkpointing | 0.73x | 0.21x (-16.1 GB) | 2048 |
| Gradient accumulation (micro-batch 64) | 0.92x | 0.20x (-16.4 GB) | 2048+ (cap, no OOM) |
| 8-bit optimiser | 0.98x | 0.97x (-0.6 GB) | 512 |
| A: bf16 + 8-bit optimiser | 2.06x | 0.65x (-7.3 GB) | 512 |
| B: bf16 + checkpointing + 8-bit optimiser | 1.62x | 0.16x (-17.3 GB) | 2048 |

- **bf16** is the only technique that makes training faster: the 4090's tensor cores run bf16 matmuls much faster than
  fp32. Activations are half size, so memory drops by about a third at large batch. Weights and optimizer stay fp32,
  so at batch 32 the saving is small (2.47 to 2.25 GB) but wall-clock falls from 22.4 s to 15.6 s.
- **Gradient checkpointing** trades compute for memory: about 27% slower, activation memory per sample falls sharply, and
  the max batch rises from 512 to 2048 (4x). It still fails at 4096.
- **Gradient accumulation** does not raise the largest micro-batch. It decouples effective batch from memory: peak stays
  flat at 4.14 GB from effective batch 256 to 2048. The sweep stopped at 2048 by choice, so no wall was found.
  The accuracy run used micro-batch 8 (4 steps of accumulation), which is inefficient: 37.1 s wall-clock versus 22.4 s.
- **8-bit optimiser** saves only the optimizer state: about 1 GB at small batches (roughly half of the 2.1 GB footprint at
  batch 8) but only 0.6 GB at batch 512, where activations dominate. It does not change the max batch for BERT-base.
  It would matter far more for a model with many more parameters.
- **Combination B** has the lowest memory (1.08 GB at batch 32, 3.22 GB at batch 512) but gives up part of the bf16 speedup.
  Its max batch (2048) equals checkpointing alone: at 2048 it uses 10.9 GB, and 4096 still fails.

## Accuracy
Test accuracy (750 rows) is 66.53% for baseline, bf16 and checkpointing, 66.67% for accumulation, 67.20% for the 8-bit
optimiser and 65.87% for both combinations. One test example is 0.13 percentage points, so the spread is within about
5 examples of the baseline and the standard error of a 750-row accuracy is roughly 1.7 points. These runs use one seed,
so **no technique is shown to help or hurt accuracy**. Identical 66.53% for three runs is a coincidence of the count
of correct predictions (499 of 750), not a copied result: bf16 has a different best validation accuracy (69.20% versus 68.53%
for baseline). Checkpointing matches the baseline's 68.53%, as expected, since it recomputes the same math and only changes
memory. Gradient accumulation differs slightly from baseline because the loss weighting of the final
partial group and the dropout random-number order change.

## Which configuration is best (depends on the goal)
- Lowest peak memory: Combination B.
- Highest throughput and fastest fine-tune at the HW4 setting: bf16 alone (2307 samples/sec at batch 512; 15.6 s for the fine-tune).
  Combination A is within 2 to 3% of it and saves another 0.6 GB.
- Largest batch: checkpointing alone or Combination B (2048), or accumulation if memory-unbounded effective batch is what matters.
The single configuration for a full fine-tune is chosen in the next step using the GPU-hour calculation.

## Caveats
- Single card, single seed. No cross-card or multi-seed claims.
- Throughput at batch sizes 1024 and above used 6 timed steps instead of 20 to keep the sweep short.
- Batches above 3,500 rows are sampled with replacement: this measures step speed and memory, not convergence.
- "Max batch" for accumulation is a sweep cap, not a measured wall.
- The 8-bit optimiser (bitsandbytes) only quantises tensors above a size threshold, so small parameters stay fp32.
