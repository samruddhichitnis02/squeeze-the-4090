"""GPU-hours for a full fine-tune: prediction from measured step throughput vs real full-data runs."""
import csv
import json
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step5"
S4 = "results/step4"
CFGS = [
    # name, label, step4 sweep dir, batch size
    ("fp32_bs32", "fp32, batch 32 (HW4 baseline)", "baseline", 32),
    ("bf16_bs32", "bf16, batch 32", "bf16", 32),
    ("bf16_bs128", "bf16, batch 128", "bf16", 128),
    ("fp32_bs128", "fp32, batch 128", "baseline", 128),
]

rows, md_rows, csv_rows = [], [], []
for name, label, sweep, bs in CFGS:
    try:
        res = json.load(open(f"{out_dir}/{name}/hw4a_results.json"))
        sps = json.load(open(f"{S4}/{sweep}/raw/bs{bs:06d}.json"))["samples_per_sec"]
    except FileNotFoundError:
        continue
    n_train, epochs = res["split"]["train"], res["settings"]["epochs"]
    samples = n_train * epochs
    predicted = samples / sps
    train_only = sum(h["epoch_seconds"] for h in res["history"])
    wall = res["train_wallclock_seconds"]
    best = max(res["history"], key=lambda h: h["val_acc"])
    rows.append(dict(name=name, label=label, n_train=n_train, epochs=epochs, samples=samples, sps=sps,
                     predicted=predicted, train_only=train_only, wall=wall, gpu_hours=wall / 3600,
                     peak=res["peak_mem_allocated_gb"], test_acc=res["test_acc"], best_epoch=best["epoch"],
                     best_val=best["val_acc"], split=res["split"]))
    md_rows.append(f"| {label} | {sps:.0f} | {predicted:.0f} | {train_only:.0f} | {wall:.0f} | "
                   f"{wall / 3600:.4f} | {res['peak_mem_allocated_gb']:.2f} | {best['epoch']} | "
                   f"{res['test_acc'] * 100:.2f}% |")
    csv_rows.append([label, round(sps, 1), round(predicted, 1), round(train_only, 1), round(wall, 1),
                     round(wall / 3600, 5), round(res["peak_mem_allocated_gb"], 3), best["epoch"],
                     round(res["test_acc"], 4)])

md = ["# GPU-hours for one full fine-tune (RTX 4090, 1 GPU)", ""]
if rows:
    r = rows[0]
    md += [f"Full fine-tune = all 45,615 train-split rows, split 70/15/15 with seed 2122: {r['split']['train']} train / "
           f"{r['split']['val']} val / {r['split']['test']} test, {r['epochs']} epochs.", "",
           "## How it is calculated", "",
           "```",
           "samples processed      = train rows x epochs",
           f"                       = {r['n_train']} x {r['epochs']} = {r['samples']}",
           "predicted seconds      = samples processed / step throughput (samples/sec, from the Step 4 sweep at that batch size)",
           "GPU-hours              = wall-clock seconds / 3600   (1 GPU, so GPU-hours = hours)",
           "```", ""]
    b = rows[0]
    md += [f"Worked example, {b['label']}: {b['samples']} / {b['sps']:.1f} = {b['predicted']:.0f} s predicted. "
           f"Measured: {b['train_only']:.0f} s of pure training plus validation after each epoch = {b['wall']:.0f} s wall-clock "
           f"= {b['wall']:.0f} / 3600 = {b['gpu_hours']:.4f} GPU-hours.", ""]
md += ["| Configuration | Step throughput (samples/s) | Predicted (s) | Measured training only (s) | Measured wall-clock incl. validation (s) | GPU-hours | Peak memory (GB) | Best epoch | Test accuracy |",
       "|---|---|---|---|---|---|---|---|---|", *md_rows, "",
       "Wall-clock is what the GPU was reserved for, so GPU-hours use it. The prediction covers training steps only, so the gap "
       "between predicted and measured-training-only shows how well step throughput from the sweep carries over to a real run "
       "(data transfer from the CPU, optimizer on the last partial batch, and the DataLoader are included in the real run).", ""]
open(f"{out_dir}/gpu_hours.md", "w").write("\n".join(md))
with open(f"{out_dir}/gpu_hours.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["configuration", "step_samples_per_sec", "predicted_s", "measured_training_s", "measured_wallclock_s",
                "gpu_hours", "peak_mem_gb", "best_epoch", "test_acc"])
    w.writerows(csv_rows)
print("\n".join(md))
