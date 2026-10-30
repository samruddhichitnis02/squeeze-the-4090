"""Build the batch-scaling table (md + csv), plots, and a fixed-vs-per-sample memory fit."""
import csv
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step3"
runs = sorted((json.load(open(p)) for p in glob.glob(f"{out_dir}/raw/*.json")), key=lambda r: r["batch_size"])
ok = [r for r in runs if r["status"] == "ok"]
oom = [r for r in runs if r["status"] == "oom"]
PLATEAU_GAIN = 0.05  # throughput "stops improving" when doubling batch gains less than 5%

md = ["# Batch scaling to the memory wall", ""]
if runs:
    r0 = runs[0]
    md.append(f"GPU: {r0['gpu']} ({r0['gpu_total_gb']:.1f} GB total). BERT fine-tune, {r0['precision']}, "
              f"max_len {r0['max_len']}, AdamW lr {r0['lr']}. {r0['timed_steps']} timed steps per batch size "
              f"after {r0['warmup_steps']} warmup steps, one fresh process per size.")
    md.append(f"Memory wall definition: PyTorch allocations are capped at {r0.get('usable_gb', 0):.1f} GB "
              f"(95% of the {r0.get('free_gb_at_start', 0):.1f} GB free at start; Windows holds the rest for the display). "
              "The cap makes overflow a real out-of-memory error, since the Windows driver would otherwise spill to system RAM.")
md.append("Batches larger than the 3,500 train rows are sampled with replacement. Epoch time is estimated as "
          "3,500 / samples-per-second (not a measured epoch).")
md += ["", "| Batch size | Status | Samples/sec | Peak memory allocated (GB) | Peak reserved (GB) | Est. epoch time (s) | Gain vs previous size |",
       "|---|---|---|---|---|---|---|"]
csv_rows, prev = [], None
plateau_at = None
for r in runs:
    if r["status"] == "ok":
        gain = "" if prev is None else f"{(r['samples_per_sec'] / prev - 1) * 100:+.1f}%"
        if prev is not None and plateau_at is None and r["samples_per_sec"] / prev - 1 < PLATEAU_GAIN:
            plateau_at = r["batch_size"]
        md.append(f"| {r['batch_size']} | ok | {r['samples_per_sec']:.1f} | {r['peak_mem_allocated_gb']:.2f} | "
                  f"{r['peak_mem_reserved_gb']:.2f} | {r['est_epoch_seconds']:.2f} | {gain} |")
        csv_rows.append([r["batch_size"], "ok", round(r["samples_per_sec"], 2), round(r["peak_mem_allocated_gb"], 3),
                         round(r["peak_mem_reserved_gb"], 3), round(r["est_epoch_seconds"], 3)])
        prev = r["samples_per_sec"]
    else:
        md.append(f"| {r['batch_size']} | **OOM** | n/a | n/a | n/a | n/a | n/a |")
        csv_rows.append([r["batch_size"], "oom", "", "", "", ""])

summary = [""]
if ok:
    mx = max(r["batch_size"] for r in ok)
    best = max(ok, key=lambda r: r["samples_per_sec"])
    summary.append(f"- Largest power-of-two batch size that runs: **{mx}**"
                   + (f" (first failure at {oom[0]['batch_size']})" if oom else " (no OOM reached; sweep cap hit)"))
    summary.append(f"- Highest throughput: {best['samples_per_sec']:.1f} samples/sec at batch {best['batch_size']}")
    if plateau_at:
        summary.append(f"- Throughput stops improving (gain under {PLATEAU_GAIN * 100:.0f}% per doubling) at batch size "
                       f"**{plateau_at}**")
    if len(ok) >= 3:
        x = np.array([r["batch_size"] for r in ok], float)
        y = np.array([r["peak_mem_allocated_gb"] for r in ok], float)
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        summary.append(f"- Linear fit of peak memory vs batch size: about {intercept:.2f} GB fixed "
                       f"(weights, gradients, optimizer states, CUDA context overhead) + {slope * 1024:.1f} MB per sample "
                       f"(max abs residual {np.abs(resid).max():.2f} GB)")
        total = ok[0].get("usable_gb", ok[0]["gpu_total_gb"])
        summary.append(f"- Fit predicts the wall near batch {((total - intercept) / slope):.0f} for the {total:.1f} GB usable cap")
md += summary
open(f"{out_dir}/scaling_table.md", "w").write("\n".join(md) + "\n")
with open(f"{out_dir}/scaling_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["batch_size", "status", "samples_per_sec", "peak_mem_allocated_gb", "peak_mem_reserved_gb", "est_epoch_seconds"])
    w.writerows(csv_rows)

if ok:
    bs = [r["batch_size"] for r in ok]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(bs, [r["samples_per_sec"] for r in ok], "o-")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("batch size")
    ax[0].set_ylabel("samples / sec")
    ax[0].set_title("Throughput vs batch size (RTX 4090)")
    if plateau_at:
        ax[0].axvline(plateau_at, ls="--", color="gray", label=f"gain < {PLATEAU_GAIN * 100:.0f}% at {plateau_at}")
        ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(bs, [r["peak_mem_allocated_gb"] for r in ok], "o-", color="tab:red")
    ax[1].axhline(ok[0].get("usable_gb", ok[0]["gpu_total_gb"]), ls=":", color="black", label="usable VRAM cap")
    if oom:
        ax[1].axvline(oom[0]["batch_size"], ls="--", color="tab:red", label=f"OOM at {oom[0]['batch_size']}")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("batch size")
    ax[1].set_ylabel("peak allocated memory (GB)")
    ax[1].set_title("Peak memory vs batch size")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/throughput_vs_batch.png", dpi=150)
print("\n".join(md))
