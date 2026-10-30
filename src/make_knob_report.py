"""Step 7 report: knob sweep tables, plots, and a roofline cross-check against measured numbers."""
import csv
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step7"
LOG_DIR = out_dir.replace("results", "logs")
ATTAIN = 0.95
CONFIGS = [
    ("base", "base (bf16, prefix cache on, chunked prefill on, 256 seqs)"),
    ("no_prefix_cache", "prefix caching off"),
    ("no_chunked_prefill", "chunked prefill off"),
    ("batched_tokens_512", "max-num-batched-tokens 512"),
    ("nocache_nochunk", "[prefix cache off] chunked prefill off"),
    ("nocache_bt512", "[prefix cache off] max-num-batched-tokens 512"),
    ("seqs_32", "max-num-seqs 32"),
    ("seqs_64", "max-num-seqs 64"),
    ("weights_fp8", "FP8 weights"),
    ("triton_kv_bf16", "[Triton attention] bf16 weights, bf16 KV (control)"),
    ("triton_kv_fp8", "[Triton attention] bf16 weights, FP8 KV cache"),
    ("triton_w8_kv_bf16", "[Triton attention] FP8 weights, bf16 KV (control)"),
    ("triton_w8_kv_fp8", "[Triton attention] FP8 weights + FP8 KV cache"),
]
NAMES = dict(CONFIGS)
REF = {"short": [96, 160], "longprefix": [32, 64]}

# Model and hardware constants for the roofline cross-check
PARAMS_BODY = 1.31e9      # Qwen2.5-1.5B transformer-layer matmul params (28 layers)
PARAMS_HEAD = 0.2334e9    # lm_head (tied with the 151,936 x 1,536 embedding)
FLOP_DECODE_TOKEN = 2 * (PARAMS_BODY + PARAMS_HEAD)   # lm_head applied to every generated token
FLOP_PREFILL_TOKEN = 2 * PARAMS_BODY                  # lm_head only on the last prompt position
PEAK_TFLOPS_BF16 = 165.6   # stock RTX 4090 FP16/BF16 tensor, FP32 accumulate, dense (TechPowerUp measurement)
PEAK_BW_GBS = 1008.0       # RTX 4090 memory bandwidth

runs = {}  # (config, workload) -> run dict
for p in sorted(glob.glob(f"{out_dir}/raw/*.json")):
    j = json.load(open(p))
    key = (j["label"], j["workload"])
    if key in runs:
        runs[key]["rates"] += j["rates"]
    else:
        runs[key] = j
for j in runs.values():
    seen, merged = set(), []
    for r in sorted(j["rates"], key=lambda r: r["offered_rate"]):
        if r["offered_rate"] not in seen:
            seen.add(r["offered_rate"])
            merged.append(r)
    j["rates"] = merged


def facts(label):
    cands = sorted(glob.glob(f"{LOG_DIR}/server_{label}*.log"))
    f = {}
    if cands:
        t = open(cands[0], errors="ignore").read()
        m = re.search(r"Model loading took ([\d.]+) GiB", t)
        f["weights_gib"] = float(m.group(1)) if m else None
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
        f["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
    return f


def at(rates, rate):
    return next((r for r in rates if abs(r["offered_rate"] - rate) < 1e-6), None)


def sla_capacity(rates):
    ok = [r for r in rates if r["sla_attainment"] >= ATTAIN and r["n_failed"] == 0]
    return max(ok, key=lambda r: r["offered_rate"]) if ok else None


md = ["# Step 7: serving knob sweep (RTX 4090, Qwen2.5-1.5B-Instruct, vLLM 0.32)", "",
      "Every configuration: bf16 unless stated, 4 API server processes, 2 client processes, max-model-len 2048, "
      "gpu-memory-utilization 0.6, 128 output tokens per request, streamed, open-loop Poisson arrivals, one run per cell.",
      "Workloads: **short** = short chat prompts (about 41 tokens, decode dominated); **longprefix** = a ~950-token document identical "
      "for every request plus a short unique question (prefill and KV-read heavy).",
      "SLA: TTFT <= 500 ms and TPOT <= 50 ms. SLA capacity = highest tested rate with >= 95% of requests inside the SLA "
      "(limited by the small rate grid). Single-run differences under roughly 15 to 20% are within noise (Step 6).", ""]
csv_rows = []
for wl, title in (("short", "Workload: short chat prompts"), ("longprefix", "Workload: long shared prefix (~950 prompt tokens)")):
    refs = REF[wl]
    md += [f"## {title}", "",
           "| Configuration | KV cache (tokens) | SLA capacity (req/s) | Peak tok/s | "
           + " | ".join(f"e2e p99 @{r} req/s (ms)" for r in refs) + " | "
           + " | ".join(f"TTFT p99 @{r} (ms)" for r in refs) + f" | TPOT p50 @{refs[0]} (ms) |",
           "|---|---|---|---|" + "---|" * (len(refs) * 2 + 1)]
    for label, name in CONFIGS:
        j = runs.get((label, wl))
        if not j:
            continue
        rates = j["rates"]
        cap = sla_capacity(rates)
        peak = max(rates, key=lambda r: r["output_tokens_per_sec"])
        cells = []
        for r in refs:
            x = at(rates, r)
            cells.append(f"{x['e2e_ms']['p99']:.0f}" if x else "n/a")
        for r in refs:
            x = at(rates, r)
            cells.append(f"{x['ttft_ms']['p99']:.0f}" if x else "n/a")
        x0 = at(rates, refs[0])
        tp = f"{x0['tpot_ms']['p50']:.1f}" if x0 else "n/a"
        f = facts(label)
        ladder_top = max(r["offered_rate"] for r in rates)
        capcell = "none"
        if cap:
            capcell = f"{cap['offered_rate']:g}" + (" (>= , ladder top)" if cap["offered_rate"] >= ladder_top else "")
        md.append(f"| {name} | {f.get('kv_tokens', 'n/a')} | {capcell} | {peak['output_tokens_per_sec']:.0f} | "
                  + " | ".join(cells) + f" | {tp} |")
        for r in rates:
            csv_rows.append([wl, label, r["offered_rate"], round(r["achieved_req_per_sec"], 2),
                             round(r["output_tokens_per_sec"], 1), round(r["e2e_ms"]["p50"], 1),
                             round(r["e2e_ms"]["p99"], 1), round(r["ttft_ms"]["p50"], 1), round(r["ttft_ms"]["p99"], 1),
                             round(r["tpot_ms"]["p50"], 2), round(r["tpot_ms"]["p99"], 2),
                             round(r["goodput_req_per_sec"], 2), r["n_failed"]])
    md.append("")

md += ["## Full per-rate results", ""]
for wl in ("short", "longprefix"):
    for label, name in CONFIGS:
        j = runs.get((label, wl))
        if not j:
            continue
        md += [f"### {name} / {wl}", "",
               "| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |",
               "|---|---|---|---|---|---|---|"]
        for r in j["rates"]:
            md.append(f"| {r['offered_rate']:g} | {r['achieved_req_per_sec']:.1f} | {r['output_tokens_per_sec']:.0f} | "
                      f"{r['e2e_ms']['p50']:.0f} / {r['e2e_ms']['p99']:.0f} | {r['ttft_ms']['p50']:.0f} / {r['ttft_ms']['p99']:.0f} | "
                      f"{r['tpot_ms']['p50']:.1f} / {r['tpot_ms']['p99']:.1f} | {r['goodput_req_per_sec']:.1f} |")
        md.append("")

md += ["## Roofline cross-check", "",
       f"Ceilings used: memory bandwidth {PEAK_BW_GBS:.0f} GB/s; bf16 tensor throughput {PEAK_TFLOPS_BF16} TFLOP/s dense with FP32 "
       "accumulate (stock RTX 4090, TechPowerUp measurement; the 82.6 TFLOPS quoted by some sites is the FP32 shader rate). "
       f"Matmul FLOPs per token = 2 x parameters: {FLOP_DECODE_TOKEN / 1e9:.2f} GFLOP per generated token, "
       f"{FLOP_PREFILL_TOKEN / 1e9:.2f} GFLOP per prompt token (the output projection runs only on the last prompt position). "
       "Attention FLOPs are ignored (small at these context lengths).", "",
       "### Decode at batch of about 1 is limited by weight reads", "",
       "Uses the Step 6 load test at 2 req/s (about 1 request in flight, e2e about 0.5 to 0.6 s), not this sweep: the lowest rate here "
       "is 32 req/s, which keeps about 20 requests in flight. Observed TPOT p50 versus the time to read the weights once at full bandwidth.", "",
       "| Configuration | Weight bytes read per step (GB, est.) | Bandwidth-bound step time (ms) | Observed TPOT p50 (ms) | Implied bandwidth use | Observed minus bound (ms) |",
       "|---|---|---|---|---|---|"]
# bf16: all matmul weights incl. lm_head at 2 bytes. FP8 weights: 112 transformer linears at 1 byte, lm_head stays bf16.
bytes_cfg = {"bf16": (PARAMS_BODY + PARAMS_HEAD) * 2, "fp8": PARAMS_BODY * 1 + PARAMS_HEAD * 2}
step6 = {}
for lab in ("bf16", "fp8"):
    p6 = f"results/step6/raw/{lab}.json"
    if os.path.exists(p6):
        step6[lab] = json.load(open(p6))
roof_rows = {}
for lab, nm in (("bf16", "bf16 weights"), ("fp8", "FP8 weights (lm_head stays bf16)")):
    if lab not in step6:
        continue
    r = next((x for x in step6[lab]["rates"] if abs(x["offered_rate"] - 2) < 1e-6), None)
    if not r:
        continue
    b = bytes_cfg[lab] / 1e9
    bound = b / PEAK_BW_GBS * 1000
    tp = r["tpot_ms"]["p50"]
    roof_rows[lab] = (bound, tp)
    md.append(f"| {nm} (Step 6, 2 req/s) | {b:.2f} | {bound:.2f} | {tp:.2f} | {bound / tp * 100:.0f}% | {tp - bound:.2f} |")
if len(roof_rows) == 2:
    pred = roof_rows["fp8"][0] / roof_rows["bf16"][0]
    obs = roof_rows["fp8"][1] / roof_rows["bf16"][1]
    md += ["", f"FP8 reads {pred * 100:.0f}% of the bf16 weight bytes, so a pure bandwidth model predicts TPOT at {pred * 100:.0f}% of bf16; "
           f"observed is {obs * 100:.0f}%. A fixed cost per step of about 1.5 to 2 ms (kernel launches, attention, sampling, norms) that "
           "does not shrink with weight bytes would explain the gap. That reading is an inference from two data points, not a profile."]
md += ["", "### Prefill throughput versus the compute ceiling (prefix caching off only)", "",
       "TFLOP/s = generated tokens x decode FLOPs + prompt tokens x prefill FLOPs, from measured rates. Prompt FLOPs are only counted "
       "when prefix caching is off, because with caching the shared prefix is computed once and the true FLOPs are far lower "
       "(a no-cache assumption would give an invalid ~100% of peak). Percent is of 165.6 TFLOP/s. Valid only where the server was "
       "saturated (the rate ladder hit its overload stop).", "",
       "| Workload | Configuration | Peak tok/s | Achieved req/s | Decode-only TFLOP/s | Decode + prefill TFLOP/s | % of bf16 peak | Saturated? |",
       "|---|---|---|---|---|---|---|---|"]
NOCACHE = ("no_prefix_cache", "nocache_nochunk", "nocache_bt512")
for wl in ("short", "longprefix"):
    for label, name in CONFIGS:
        j = runs.get((label, wl))
        if not j:
            continue
        best = max(j["rates"], key=lambda r: r["output_tokens_per_sec"])
        dec = best["output_tokens_per_sec"] * FLOP_DECODE_TOKEN / 1e12
        top = max(r["offered_rate"] for r in j["rates"])
        sat = "yes" if best["achieved_req_per_sec"] < 0.9 * best["offered_rate"] or best["offered_rate"] < top else "no (ladder top)"
        if label in NOCACHE:
            pre = best["achieved_req_per_sec"] * best["mean_prompt_tokens"] * FLOP_PREFILL_TOKEN / 1e12
            md.append(f"| {wl} | {name} | {best['output_tokens_per_sec']:.0f} | {best['achieved_req_per_sec']:.1f} | {dec:.1f} | "
                      f"{dec + pre:.1f} | {(dec + pre) / PEAK_TFLOPS_BF16 * 100:.0f}% | {sat} |")
        else:
            md.append(f"| {wl} | {name} | {best['output_tokens_per_sec']:.0f} | {best['achieved_req_per_sec']:.1f} | {dec:.1f} | "
                      f"n/a (prefix cached or short prompts) | n/a | {sat} |")
md.append("")
open(f"{out_dir}/knob_report.md", "w").write("\n".join(md))
with open(f"{out_dir}/knob_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["workload", "config", "offered_req_per_sec", "achieved_req_per_sec", "output_tok_per_sec", "e2e_p50_ms",
                "e2e_p99_ms", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "goodput_req_per_sec", "failed"])
    w.writerows(csv_rows)

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
palette = plt.cm.tab20.colors  # 20 distinct colours so no two configurations share one
colour = {label: palette[i % len(palette)] for i, (label, _) in enumerate(CONFIGS)}
for row, wl in enumerate(("short", "longprefix")):
    for label, name in CONFIGS:
        j = runs.get((label, wl))
        if not j:
            continue
        x = [r["offered_rate"] for r in j["rates"]]
        ls = "--" if label.startswith("triton") else "-"
        ax[row][0].plot(x, [r["ttft_ms"]["p99"] for r in j["rates"]], "o" + ls, color=colour[label], label=label, ms=4)
        ax[row][1].plot(x, [r["output_tokens_per_sec"] for r in j["rates"]], "o" + ls, color=colour[label], label=label, ms=4)
    ax[row][0].set_yscale("log")
    ax[row][0].set_ylabel("TTFT p99 (ms)")
    ax[row][1].set_ylabel("output tokens / sec")
    for c in (0, 1):
        ax[row][c].set_xscale("log", base=2)
        ax[row][c].set_xlabel("offered rate (req/s)")
        ax[row][c].set_title(f"{wl}: " + ("TTFT p99" if c == 0 else "output throughput"), fontsize=10)
        ax[row][c].grid(alpha=0.3)
ax[0][1].legend(fontsize=7, loc="upper left")
ax[1][1].legend(fontsize=7, loc="upper left")
fig.text(0.5, 0.005, "dashed lines = Triton attention backend runs (FP8 KV cache needs it); solid = default FlashAttention", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.015, 1, 1))
fig.savefig(f"{out_dir}/knob_curves.png", dpi=150)
print("\n".join(md[:40]))
