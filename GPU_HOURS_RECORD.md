# GPU usage record

Single card: NVIDIA GeForce RTX 4090 (own system). Script windows are taken from the 'run started' / 'run finished' timestamps in `logs/stepN/run_info.txt`.

| Step | What ran | Start | Finish | Wall-clock (min) | GPU-hours |
|---|---|---|---|---|---|
| step1 | BERT fine-tune on the 5,000-row subset | 2026-10-29 17:07 | 17:07 | 0.6 | 0.010 |
| step2 | HF pipeline vs vLLM serving benchmark | 2026-10-29 17:35 | 17:44 | 8.9 | 0.148 |
| step3 | Batch-size scaling to the memory wall | 2026-10-29 18:24 | 18:25 | 1.2 | 0.021 |
| step4 | Memory techniques, one at a time | 2026-10-29 18:47 | 19:01 | 13.5 | 0.225 |
| step5 | Full fine-tune GPU-hours check (4 configurations) | 2026-10-29 19:17 | 19:25 | 7.7 | 0.128 |
| **Scripted total** | | | | **31.9** | **0.532** |

## Not captured by the run scripts (approximate)

- **step2**: Smoke tests of vLLM startup (several short runs, each a few minutes at most, three of them failed on environment problems: missing C compiler, nvcc/header mismatch). GPU time not logged; approx 5 to 10 minutes total.
- **step2**: vLLM demo: first attempt failed immediately (missing __main__ guard), second attempt succeeded (two models, about 2.4 s generation each plus model load). Not timestamped; approx 3 minutes.
- **step3**: First sweep attempt stalled at batch 1024 because the Windows driver spilled GPU memory to system RAM (approx 22 minutes of GPU time at 100% utilization before it was stopped). Its logs were overwritten by the corrected rerun and its results were discarded.
- **step4**: Baseline regression check of the edited train_bert.py: one call hit a 10-minute tool timeout (cause was the output pipe, GPU use during that time not verified, up to 10 minutes), the rerun finished in about 30 s and reproduced Step 1.
- **step4**: bitsandbytes 8-bit optimizer check (scripts/bnb_check.py): a few seconds.
- **step5**: First launch failed within 15 s on a command-line flag typo (no GPU work); fixed and rerun.

These unlogged items add roughly 0.5 to 0.75 GPU-hours on top of the scripted total (about 22 minutes for the stalled sweep, plus up to about 25 minutes of smoke tests, demo attempts and checks). They are estimates, not measurements.

## Reservation

The GPU is the user's own RTX 4090, so no reservation was needed. See `RESERVATION.md`.
