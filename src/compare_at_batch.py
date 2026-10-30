"""Like-for-like view: every technique at the same batch sizes (from the Step 4 sweeps)."""
import glob
import json
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step4"
names = ["baseline", "bf16", "grad_ckpt", "grad_accum", "adam8bit", "combo_bf16_adam8bit", "combo_bf16_ckpt_adam8bit"]
sizes = [64, 256, 512, 1024, 2048]
data = {}
for n in names:
    for p in glob.glob(f"{out_dir}/{n}/raw/*.json"):
        r = json.load(open(p))
        data[(n, r["batch_size"])] = r

lines = ["# Like-for-like comparison at fixed batch sizes", "",
         "Each cell: samples/sec | peak memory GB (OOM = out of memory, - = not run).", "",
         "| Technique | " + " | ".join(f"batch {s}" for s in sizes) + " |",
         "|---|" + "---|" * len(sizes)]
for n in names:
    cells = []
    for s in sizes:
        r = data.get((n, s))
        if r is None:
            cells.append("-")
        elif r["status"] == "oom":
            cells.append("OOM")
        else:
            cells.append(f"{r['samples_per_sec']:.0f} \\| {r['peak_mem_allocated_gb']:.2f}")
    lines.append(f"| {n} | " + " | ".join(cells) + " |")

lines += ["", "## Ratios vs baseline at batch 512 (where both ran)", ""]
b = data.get(("baseline", 512))
for n in names[1:]:
    r = data.get((n, 512))
    if b and r and r["status"] == "ok":
        lines.append(f"- {n}: throughput {r['samples_per_sec'] / b['samples_per_sec']:.2f}x, "
                     f"memory {r['peak_mem_allocated_gb'] / b['peak_mem_allocated_gb']:.2f}x "
                     f"({r['peak_mem_allocated_gb'] - b['peak_mem_allocated_gb']:+.2f} GB)")
open(f"{out_dir}/like_for_like.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
