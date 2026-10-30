# GPU reservation record

No reservation was needed. All runs used the author's own workstation, so there was no booking or queue.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (24,564 MiB reported by nvidia-smi) |
| GPU UUID | GPU-c5ad1a63-b7c0-1838-5d8f-0df577ec82ce (also in `logs/step*/gpu_uuid.txt`) |
| Driver | NVIDIA 610.60 (CUDA runtime up to 13.3 supported) |
| Host | AMD Ryzen 9 7950X, 128 GB RAM |
| OS | Windows with WSL2, Ubuntu 26.04 LTS |
| Access | Exclusive use of the card. Windows kept about 1.6 GB of VRAM for the display, which is why the usable memory in the sweeps is capped at 21.3 GB |
| Dates of use | 2026-10-29 (all scripted runs) |

Only one card was available. The second card the assignment asks for was not used, so the two-card UUID capture and
the cross-card comparisons are not part of these results.
