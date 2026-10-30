"""Build the serving comparison table (markdown + csv) from raw per-config JSON files."""
import csv
import glob
import json
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step2"
rows = [json.load(open(p)) for p in glob.glob(f"{out_dir}/raw/*.json")]
order = {("hf", 1): 0, ("vllm", 1): 1, ("vllm", 8): 2, ("vllm", 32): 3, ("hf", 8): 4, ("hf", 32): 5}
rows.sort(key=lambda r: order.get((r["engine"], r["batch_size"]), 99))
base = next((r for r in rows if r["engine"] == "hf" and r["batch_size"] == 1), None)


def name(r):
    label = "HF pipeline" if r["engine"] == "hf" else "vLLM"
    extra = " (extra, not required)" if r["engine"] == "hf" and r["batch_size"] > 1 else ""
    return f"{label}, batch {r['batch_size']}{extra}"


md = ["# Serving comparison", ""]
if rows:
    md.append(f"Model: {rows[0]['model']}, bf16, greedy, {rows[0]['max_new_tokens']} new tokens per request "
              f"(EOS suppressed), {rows[0]['num_batches']} timed calls per configuration after warmup.")
md += ["", "Latency is per generate call. At batch 1 that is one request; at batch 8/32 it is the whole batch.", ""]
md.append("| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) | Requests/sec | Throughput vs HF b1 | p50 latency vs HF b1 |")
md.append("|---|---|---|---|---|---|---|")
csv_rows = []
for r in rows:
    l = r["latency_ms"]
    tp = f"{r['tokens_per_sec'] / base['tokens_per_sec']:.2f}x" if base else "n/a"
    lt = f"{l['p50'] / base['latency_ms']['p50']:.2f}x" if base else "n/a"
    md.append(f"| {name(r)} | {r['tokens_per_sec']:.1f} | {l['p50']:.1f} | {l['p95']:.1f} | "
              f"{r['requests_per_sec']:.2f} | {tp} | {lt} |")
    csv_rows.append([name(r), round(r["tokens_per_sec"], 2), round(l["p50"], 2), round(l["p95"], 2),
                     round(r["requests_per_sec"], 3)])
open(f"{out_dir}/serving_table.md", "w").write("\n".join(md) + "\n")
with open(f"{out_dir}/serving_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["configuration", "tokens_per_sec", "p50_latency_ms", "p95_latency_ms", "requests_per_sec"])
    w.writerows(csv_rows)
print("\n".join(md))
