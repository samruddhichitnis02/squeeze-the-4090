"""Build Table HW4.5.1 (md + csv), an extended table, and comparison plots from the Step 4 runs."""
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step4"
TECH = [
    ("baseline", "Baseline (HW4 settings)"),
    ("bf16", "Mixed precision (bf16)"),
    ("grad_ckpt", "Gradient checkpointing"),
    ("grad_accum", "Gradient accumulation (micro-batch 64)"),
    ("adam8bit", "8-bit optimiser"),
    ("combo_bf16_adam8bit", "Combination A: bf16 + 8-bit optimiser"),
    ("combo_bf16_ckpt_adam8bit", "Combination B: bf16 + checkpointing + 8-bit optimiser"),
]
EPOCHS_NOTE = "4 epochs, batch 32, seed 2122 (same as the baseline fine-tune)"


def load(name):
    runs = sorted((json.load(open(p)) for p in glob.glob(f"{out_dir}/{name}/raw/*.json")),
                  key=lambda r: r["batch_size"])
    q = f"{out_dir}/{name}/quality/hw4a_results.json"
    return runs, (json.load(open(q)) if os.path.exists(q) else None)


rows, ext, csv_rows, series = [], [], [], {}
for name, label in TECH:
    runs, q = load(name)
    ok = [r for r in runs if r["status"] == "ok"]
    oom = [r for r in runs if r["status"] == "oom"]
    if not ok:
        continue
    series[label] = ok
    top = ok[-1]  # largest batch that ran
    wall = f"{top['batch_size']}" if oom else f"{top['batch_size']}+ (no OOM; sweep cap)"
    acc = f"{q['test_acc'] * 100:.2f}%" if q else "n/a"
    rows.append(f"| {label} | {wall} | {top['samples_per_sec']:.1f} | {top['peak_mem_allocated_gb']:.2f} | {acc} |")
    csv_rows.append([label, top["batch_size"], "oom_reached" if oom else "cap_reached",
                     round(top["samples_per_sec"], 2), round(top["peak_mem_allocated_gb"], 3),
                     round(q["test_acc"], 4) if q else ""])
    if q:
        n_samples = q["split"]["train"] * q["settings"]["epochs"]
        ext.append(f"| {label} | {q['peak_mem_allocated_gb']:.2f} | {q['train_wallclock_seconds']:.1f} | "
                   f"{n_samples / q['train_wallclock_seconds']:.0f} | {q['best_val_acc_epoch']} | "
                   f"{max(h['val_acc'] for h in q['history']) * 100:.2f}% | {q['test_acc'] * 100:.2f}% |")

md = ["# Table HW4.5.1: memory techniques measured one at a time (RTX 4090)", "",
      "Max batch = largest power-of-two batch that completed the sweep (fresh process per size, allocator capped "
      "at the usable VRAM so overflow is a real OOM). For gradient accumulation the number is the largest "
      "effective batch tested with micro-batch 64; it is not limited by memory.",
      "Throughput and peak memory are measured at that max batch. Eval accuracy is test accuracy after a real "
      f"fine-tune at the HW4 setting: {EPOCHS_NOTE}, best-validation-accuracy epoch.", "",
      "| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |",
      "|---|---|---|---|---|", *rows, ""]
md += ["# Extended table: the HW4 fine-tune itself (batch 32, 4 epochs)", "",
       "Training samples/sec = 3,500 x 4 epochs / wall-clock, so it includes per-epoch validation time.", "",
       "| Technique | Peak memory (GB) | Train wall-clock (s) | Training samples/sec | Best epoch | Best val acc | Test acc |",
       "|---|---|---|---|---|---|---|", *ext, ""]
open(f"{out_dir}/technique_table.md", "w").write("\n".join(md))
with open(f"{out_dir}/technique_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["technique", "max_batch_ran", "sweep_end", "samples_per_sec_at_max", "peak_mem_gb_at_max", "test_acc"])
    w.writerows(csv_rows)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for label, ok in series.items():
    bs = [r["batch_size"] for r in ok]
    ax[0].plot(bs, [r["samples_per_sec"] for r in ok], "o-", label=label)
    ax[1].plot(bs, [r["peak_mem_allocated_gb"] for r in ok], "o-", label=label)
for a_, t, yl in ((ax[0], "Throughput vs batch size", "samples / sec"), (ax[1], "Peak memory vs batch size", "peak allocated (GB)")):
    a_.set_xscale("log", base=2)
    a_.set_xlabel("batch size (effective)")
    a_.set_ylabel(yl)
    a_.set_title(t)
    a_.grid(alpha=0.3)
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{out_dir}/technique_comparison.png", dpi=150)
print("\n".join(md))
