"""Step 8 report: quality (paired against bf16) next to speed and memory, with a run-to-run noise floor."""
import csv
import glob
import gzip
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step8"
LOG_DIR = out_dir.replace("results", "logs")
REF, REF2 = "bf16_ref", "bf16_ref2"
ORDER = [
    ("bf16_ref", "bf16 (reference)"),
    ("bf16_ref2", "bf16 repeat (noise floor)"),
    ("fp16", "fp16"),
    ("fp8_weights", "FP8 weights"),
    ("triton_bf16", "[Triton] bf16 (control)"),
    ("triton_fp8kv", "[Triton] FP8 KV, e4m3 fixed scale"),
    ("triton_fp8kv_e5m2", "[Triton] FP8 KV, e5m2"),
    ("triton_fp8kv_pth", "[Triton] FP8 KV, per-token-head scale"),
    ("triton_w8_fp8kv", "[Triton] FP8 weights + FP8 KV (e4m3)"),
    ("awq_int4", "INT4 AWQ (weight-only)"),
    ("gptq_int4", "INT4 GPTQ (weight-only)"),
]
NAMES = dict(ORDER)

R = {}
for p in glob.glob(f"{out_dir}/raw/*.json"):
    j = json.load(open(p))
    R[j["label"]] = j
if REF not in R:
    sys.exit("reference run missing")


def bits(s):
    return [int(c) for c in s]


def paired(label, key):
    a, b = bits(R[label][key]["correct"]), bits(R[REF][key]["correct"])
    if len(a) != len(b):
        return None
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    better = sum(1 for x in d if x > 0)
    worse = sum(1 for x in d if x < 0)
    return dict(diff=mean, se=se, better=better, worse=worse)


def acc_ci(j, key):
    n, p = j[key]["n"], j[key]["accuracy"]
    return p, 1.96 * math.sqrt(p * (1 - p) / n)


def outputs(label):
    p = f"{out_dir}/raw/{label}.gsm8k_outputs.json.gz"
    return json.load(gzip.open(p, "rt", encoding="utf-8")) if os.path.exists(p) else None


def identical_rate(label):
    a, b = outputs(label), outputs(REF)
    if not a or not b or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def server_facts(label):
    f = {}
    p = f"{LOG_DIR}/eval_{label}.log"
    if os.path.exists(p):
        t = open(p, errors="ignore").read()
        m = re.search(r"Model loading took ([\d.]+) GiB", t)
        f["weights_gib"] = float(m.group(1)) if m else None
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
        f["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Using (\w+) backend|Using AttentionBackendEnum\.(\w+) backend", t)
        f["backend"] = (m.group(1) or m.group(2)) if m else None
    return f


def verdict(d, noise_se):
    if d is None:
        return "n/a"
    z = abs(d["diff"]) / d["se"] if d["se"] > 0 else 0
    if z < 2:
        return "within noise"
    return ("better" if d["diff"] > 0 else "worse") + f" ({z:.1f} SE)"


ref = R[REF]
ref_speed = ref["speed"]
rows = []
for label, name in ORDER:
    if label not in R:
        continue
    j = R[label]
    f = server_facts(label)
    mm, mm_ci = acc_ci(j, "mmlu")
    gs, gs_ci = acc_ci(j, "gsm8k")
    rows.append(dict(
        label=label, name=name, ppl=j["ppl"]["perplexity"], mmlu=mm, mmlu_ci=mm_ci, gsm=gs, gsm_ci=gs_ci,
        d_mmlu=paired(label, "mmlu") if label != REF else None, d_gsm=paired(label, "gsm8k") if label != REF else None,
        identical=identical_rate(label) if label != REF else None,
        ms_tok=j["speed"]["batch1_ms_per_token"], tps256=j["speed"]["batch256_tokens_per_sec"],
        weights=f.get("weights_gib"), kv=f.get("kv_tokens"), backend=f.get("backend") or j.get("attention_backend")))

md = ["# Step 8: quantisation, speed and measured quality (RTX 4090, Qwen2.5-1.5B-Instruct, vLLM 0.32)", "",
      f"Quality sets: WikiText-2 test perplexity ({ref['ppl']['tokens_scored']:,} tokens, window {ref['ppl']['window']}), "
      f"MMLU zero-shot all subjects (n={ref['mmlu']['n']:,}), GSM8K zero-shot chain-of-thought (n={ref['gsm8k']['n']:,}, greedy, 512 tokens max).",
      "Differences are **paired against the bf16 reference question by question**; the standard error (SE) comes from the per-question differences. "
      "\"Within noise\" means the paired difference is under 2 SE. The row \"bf16 repeat\" runs the identical configuration a second time and shows what "
      "pure run-to-run variation looks like.", ""]

md += ["## Quality", "",
       "| Configuration | Perplexity | MMLU (95% CI) | MMLU diff vs bf16 (paired) | GSM8K (95% CI) | GSM8K diff vs bf16 (paired) | Questions better / worse (GSM8K) | GSM8K outputs identical to bf16 |",
       "|---|---|---|---|---|---|---|---|"]
for r in rows:
    dm, dg = r["d_mmlu"], r["d_gsm"]
    md.append(f"| {r['name']} | {r['ppl']:.3f} | {r['mmlu'] * 100:.2f}% (±{r['mmlu_ci'] * 100:.2f}) | "
              + (f"{dm['diff'] * 100:+.2f} pts, {verdict(dm, 0)}" if dm else "reference") + f" | {r['gsm'] * 100:.2f}% (±{r['gsm_ci'] * 100:.2f}) | "
              + (f"{dg['diff'] * 100:+.2f} pts, {verdict(dg, 0)}" if dg else "reference") + " | "
              + (f"{dg['better']} / {dg['worse']}" if dg else "n/a") + " | "
              + (f"{r['identical'] * 100:.1f}%" if r["identical"] is not None else "n/a") + " |")

md += ["", "## Speed and memory (same processes)", "",
       "Batch-1 time per output token is the median of 12 single-request calls of 128 tokens. Batch-256 throughput is the median of 4 engine-only runs "
       "(no HTTP layer) with 256 concurrent sequences. Weights are what vLLM reports as loaded; KV capacity is tokens that fit in the 0.6 memory budget.", "",
       "| Configuration | Attention backend | Weights (GiB) | KV cache (tokens) | Batch-1 ms/token | vs bf16 | Batch-256 tokens/s | vs bf16 |",
       "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['name']} | {r['backend']} | {r['weights'] if r['weights'] else 'n/a'} | {r['kv']:,} | {r['ms_tok']:.2f} | "
              f"{r['ms_tok'] / ref_speed['batch1_ms_per_token']:.2f}x | {r['tps256']:,.0f} | {r['tps256'] / ref_speed['batch256_tokens_per_sec']:.2f}x |"
              if r["kv"] else f"| {r['name']} | {r['backend']} | n/a | n/a | {r['ms_tok']:.2f} | | {r['tps256']:,.0f} | |")

# Noise floor
nf = next((r for r in rows if r["label"] == REF2), None)
if nf:
    md += ["", "## Run-to-run noise floor (bf16 vs bf16, identical settings)", "",
           f"GSM8K accuracy difference {nf['d_gsm']['diff'] * 100:+.2f} pts (SE {nf['d_gsm']['se'] * 100:.2f}); MMLU difference {nf['d_mmlu']['diff'] * 100:+.2f} pts "
           f"(SE {nf['d_mmlu']['se'] * 100:.2f}); GSM8K outputs identical for {nf['identical'] * 100:.1f}% of problems; batch-1 ms/token "
           f"{nf['ms_tok']:.2f} vs {ref_speed['batch1_ms_per_token']:.2f}; batch-256 tokens/s {nf['tps256']:,.0f} vs {ref_speed['batch256_tokens_per_sec']:,.0f}. "
           "Any configuration difference smaller than this spread is not evidence of an effect."]

open(f"{out_dir}/quant_table.md", "w").write("\n".join(md) + "\n")
with open(f"{out_dir}/quant_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["config", "perplexity", "mmlu", "gsm8k", "mmlu_diff_pts", "mmlu_diff_se_pts", "gsm8k_diff_pts", "gsm8k_diff_se_pts",
                "gsm8k_identical_to_bf16", "batch1_ms_per_token", "batch256_tokens_per_sec", "weights_gib", "kv_tokens"])
    for r in rows:
        dm, dg = r["d_mmlu"], r["d_gsm"]
        w.writerow([r["name"], round(r["ppl"], 4), round(r["mmlu"], 4), round(r["gsm"], 4),
                    round(dm["diff"] * 100, 3) if dm else "", round(dm["se"] * 100, 3) if dm else "",
                    round(dg["diff"] * 100, 3) if dg else "", round(dg["se"] * 100, 3) if dg else "",
                    round(r["identical"], 4) if r["identical"] is not None else "", round(r["ms_tok"], 3), round(r["tps256"], 1),
                    r["weights"], r["kv"]])

# Figures
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
labs = [r["name"].replace(" (", "\n(") for r in rows]
y = list(range(len(rows)))
for a_, key, title in ((ax[0], "d_mmlu", "MMLU: change vs bf16 (percentage points)"), (ax[1], "d_gsm", "GSM8K: change vs bf16 (percentage points)")):
    vals = [(r[key]["diff"] * 100 if r[key] else 0) for r in rows]
    errs = [(1.96 * r[key]["se"] * 100 if r[key] else 0) for r in rows]
    cols = ["#7f7f7f" if r[key] is None else ("#2ca02c" if abs(r[key]["diff"]) < 2 * r[key]["se"] else "#d62728") for r in rows]
    a_.barh(y, vals, xerr=errs, color=cols, capsize=3)
    a_.axvline(0, color="black", lw=1)
    a_.set_yticks(y)
    a_.set_yticklabels(labs if a_ is ax[0] else [""] * len(rows), fontsize=7)
    a_.invert_yaxis()
    a_.set_title(title, fontsize=10)
    a_.set_xlabel("green = within noise, red = outside 2 SE (bars: 95% CI of the paired difference)", fontsize=7)
ax[2].scatter([r["ms_tok"] for r in rows], [r["gsm"] * 100 for r in rows], s=[(r["weights"] or 3) * 60 for r in rows], alpha=0.7)
for r in rows:
    ax[2].annotate(r["label"], (r["ms_tok"], r["gsm"] * 100), fontsize=6, xytext=(4, 4), textcoords="offset points")
ax[2].set_xlabel("batch-1 time per output token (ms, lower is faster)")
ax[2].set_ylabel("GSM8K accuracy (%)")
ax[2].set_title("Speed vs quality (bubble area = weights size)", fontsize=10)
ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{out_dir}/quant_overview.png", dpi=140)
print("\n".join(md))
