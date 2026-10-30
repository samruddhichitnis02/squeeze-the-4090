# Batch scaling analysis (single card: RTX 4090)

Data: `scaling_table.md` / `scaling_table.csv`, plot `throughput_vs_batch.png`, raw runs in `raw/`, logs in `logs/step3/`.
Setup: Step 1 fine-tune unchanged (BERT-base, fp32, max_len 64, AdamW lr 2e-5), 20 timed steps per size,
one fresh process per size, PyTorch allocations capped at 21.3 GB (95% of the VRAM free at start).

## Results
- Largest power-of-two batch size that runs: 512 (20.5 GB peak, 1081 samples/sec). Batch 1024 fails with a real CUDA
  out-of-memory error ("tried to allocate 768 MiB", 20.62 GiB already allocated).
- Throughput rises from 307 samples/sec (batch 8) to 980 (batch 128), then adds only +4.5% (256) and +5.6% (512).
  The 5%-per-doubling rule flags 256 as the point where throughput stops improving, but the curve is flat from about
  128 onward (980, 1025, 1081), so the practical knee is 128 to 256. Beyond that, extra batch size mostly costs memory.
- Peak memory is about 1.45 GB fixed plus 38 MB per sample (linear fit). The fit is rough at small batches, where
  memory is flat near 2.1 GB (weights, gradients and Adam states already take about 1.8 GB), and good at large batches
  (256 to 512 gives about 37 MB per sample). The fit predicts a wall near batch 534, consistent with 512 passing and 1024 failing.
- Sanity check: batch 32 gives 679 samples/sec here versus 673 in the Step 1 training run.

## Cross-card comparison (not measured)
The assignment asks for the max-batch ratio between two cards compared with their VRAM ratio. Only one card was used, so
no ratio was measured and none is claimed. The expected reason the two ratios differ: memory has a fixed part that does not
shrink with batch size (weights, gradients, optimizer states, CUDA context, framework overhead) and a per-sample part
(activations). Only the memory left after the fixed part is available for activations, so max batch scales with
(VRAM minus fixed) and not with VRAM itself. The display/driver reservation and allocator fragmentation also differ by card,
and power-of-two stepping adds rounding, so a given VRAM ratio need not show up as the same max-batch ratio.

## Caveats
- Batches above 3,500 rows are sampled with replacement. This measures step speed and memory, not convergence.
- Epoch times are estimates (3,500 / samples per second), not measured epochs.
- The wall depends on the cap: Windows holds part of the 24 GB for the display, so usable VRAM is about 22.4 GB.
- Without the cap, the Windows driver spilled to system RAM at batch 1024 and the run stalled instead of failing (I stopped it
  after about 22 minutes). That first attempt is not part of these results.
