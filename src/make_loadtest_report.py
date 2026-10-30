"""Tables and plots for the online load test, plus memory facts parsed from the vLLM server logs."""
import csv
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step6"
LOG_DIR = out_dir.replace("results", "logs")
GPU_USD_PER_HOUR = 0.50  # ASSUMPTION for illustration only; the GPU is owned. GPU-seconds per 1M tokens is price-free.
ATTAIN = 0.95  # SLA capacity = highest offered rate where at least 95% of requests met the TTFT and TPOT targets
ORDER = ["bf16", "fp16", "fp8"]

runs = {}
for p in sorted(glob.glob(f"{out_dir}/raw/*.json")):
    j = json.load(open(p))
    if j["label"] in runs:  # refinement pass for the same precision: merge its rates into the main run
        runs[j["label"]]["rates"] += j["rates"]
    else:
        runs[j["label"]] = j
for j in runs.values():
    seen, merged = set(), []
    for r in sorted(j["rates"], key=lambda r: r["offered_rate"]):
        if r["offered_rate"] not in seen:
            seen.add(r["offered_rate"])
            merged.append(r)
    j["rates"] = merged
labels = [l for l in ORDER if l in runs] + sorted(l for l in runs if l not in ORDER)


def server_facts(label):
    p = f"{LOG_DIR}/server_{label}.log"
    facts = {}
    if os.path.exists(p):
        t = open(p, errors="ignore").read()
        m = re.search(r"Model loading took ([\d.]+) GiB", t)
        facts["weights_gib"] = float(m.group(1)) if m else None
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
        facts["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", t)
        facts["max_concurrency"] = float(m.group(1)) if m else None
    return facts


md = ["# Online load test: vLLM serving at bf16, fp16 and fp8 (RTX 4090)", ""]
if labels:
    r0 = runs[labels[0]]
    s0 = r0["rates"][0]["sla"]
    md += [f"Model {r0['model']}, {r0['max_tokens']} output tokens per request (EOS suppressed), streaming, open-loop Poisson arrivals "
           f"(seed {r0['seed']}), prompts cycle through the same 32 prompts as the offline benchmark.",
           f"SLA used for goodput: TTFT <= {s0['ttft_ms']:.0f} ms and TPOT <= {s0['tpot_ms']:.0f} ms per request. "
           f"SLA capacity = highest offered rate where >= {ATTAIN * 100:.0f}% of requests met it.", ""]

summary_rows, csv_rows = [], []
for l in labels:
    rates = runs[l]["rates"]
    md += [f"## {l}", "",
           "| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p95 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) | Failed |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rates:
        e, t, o = r["e2e_ms"], r["ttft_ms"], r["tpot_ms"]
        md.append(f"| {r['offered_rate']:g} | {r['achieved_req_per_sec']:.1f} | {r['output_tokens_per_sec']:.0f} | "
                  f"{e['p50']:.0f} / {e['p95']:.0f} / {e['p99']:.0f} | {t['p50']:.0f} / {t['p99']:.0f} | "
                  f"{o['p50']:.1f} / {o['p99']:.1f} | {r['goodput_req_per_sec']:.1f} | {r['n_failed']} |")
        csv_rows.append([l, r["offered_rate"], round(r["achieved_req_per_sec"], 2), round(r["output_tokens_per_sec"], 1),
                         round(e["p50"], 1), round(e["p95"], 1), round(e["p99"], 1), round(t["p50"], 1), round(t["p99"], 1),
                         round(o["p50"], 2), round(o["p99"], 2), round(r["goodput_req_per_sec"], 2), r["n_failed"]])
    md.append("")
    peak = max(rates, key=lambda r: r["output_tokens_per_sec"])
    sla_ok = [r for r in rates if r["sla_attainment"] >= ATTAIN and r["n_failed"] == 0]
    cap = max(sla_ok, key=lambda r: r["offered_rate"]) if sla_ok else None
    f = server_facts(l)
    summary_rows.append((l, peak, cap, f))

md += ["## Summary", "",
       f"Cost columns: GPU-seconds per 1M output tokens = 1e6 / tok/s (no price needed). Dollar cost uses an ASSUMED "
       f"${GPU_USD_PER_HOUR:.2f} per GPU-hour, which is an illustrative number, not a measured or quoted price.", "",
       "SLA capacity is only as precise as the tested rate grid (see the per-precision tables for the rates tried).", "",
       "| Precision | Peak output tok/s (offered req/s) | GPU-s per 1M tokens at peak | SLA capacity (req/s) | tok/s at SLA capacity | "
       "e2e p99 at SLA capacity (ms) | GPU-s per 1M tokens at SLA capacity | $ per 1M tokens at SLA capacity (assumed price) | "
       "Weights (GiB) | KV cache (tokens) |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for l, peak, cap, f in summary_rows:
    if cap:
        gps = 1e6 / cap["output_tokens_per_sec"]
        cap_cells = (f"{cap['offered_rate']:g} | {cap['output_tokens_per_sec']:.0f} | {cap['e2e_ms']['p99']:.0f} | "
                     f"{gps:.1f} | ${gps / 3600 * GPU_USD_PER_HOUR:.4f}")
    else:
        cap_cells = "none | n/a | n/a | n/a | n/a"
    md.append(f"| {l} | {peak['output_tokens_per_sec']:.0f} ({peak['offered_rate']:g}) | "
              f"{1e6 / peak['output_tokens_per_sec']:.1f} | {cap_cells} | "
              f"{f.get('weights_gib', 'n/a')} | {f.get('kv_tokens', 'n/a')} |")
md.append("")
open(f"{out_dir}/loadtest_table.md", "w").write("\n".join(md))
with open(f"{out_dir}/loadtest_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["precision", "offered_req_per_sec", "achieved_req_per_sec", "output_tok_per_sec", "e2e_p50_ms", "e2e_p95_ms",
                "e2e_p99_ms", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "goodput_req_per_sec", "failed"])
    w.writerows(csv_rows)

if labels:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for l in labels:
        rs = runs[l]["rates"]
        x = [r["offered_rate"] for r in rs]
        ax[0].plot(x, [r["output_tokens_per_sec"] for r in rs], "o-", label=l)
        ax[1].plot(x, [r["e2e_ms"]["p99"] for r in rs], "o-", label=l)
        ax[2].plot(x, [r["goodput_req_per_sec"] for r in rs], "o-", label=l)
    for a_, t, yl in ((ax[0], "Output throughput vs offered load", "output tokens / sec"),
                      (ax[1], "p99 end-to-end latency vs offered load", "ms"),
                      (ax[2], "SLA goodput vs offered load", "requests / sec meeting SLA")):
        a_.set_xscale("log", base=2)
        a_.set_xlabel("offered rate (req/s)")
        a_.set_ylabel(yl)
        a_.set_title(t, fontsize=10)
        a_.grid(alpha=0.3)
        a_.legend()
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/loadtest_curves.png", dpi=150)
print("\n".join(md))
