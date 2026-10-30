"""Summary figures for the README, built from saved results (nothing typed in by hand)."""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/img"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def serving_chart():
    runs = {}
    for p in glob.glob("results/step2/raw/*.json"):
        j = json.load(open(p))
        runs[(j["engine"], j["batch_size"])] = j
    order = [("hf", 1), ("vllm", 1), ("vllm", 8), ("vllm", 32), ("hf", 8), ("hf", 32)]
    names = ["HF\nbatch 1\n(baseline)", "vLLM\nbatch 1", "vLLM\nbatch 8", "vLLM\nbatch 32", "HF\nbatch 8\n(extra)", "HF\nbatch 32\n(extra)"]
    cols = ["#7f7f7f", "#2ca02c", "#2ca02c", "#2ca02c", "#bbbbbb", "#bbbbbb"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    tps = [runs[k]["tokens_per_sec"] for k in order]
    p50 = [runs[k]["latency_ms"]["p50"] for k in order]
    p95 = [runs[k]["latency_ms"]["p95"] for k in order]
    b = ax[0].bar(range(6), tps, color=cols)
    ax[0].set_xticks(range(6))
    ax[0].set_xticklabels(names, fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("generated tokens / second (log scale)")
    ax[0].set_title("Throughput: higher is better")
    for r, v in zip(b, tps):
        ax[0].text(r.get_x() + r.get_width() / 2, v * 1.08, f"{v:,.0f}", ha="center", fontsize=8)
    ax[1].bar(range(6), p50, color=cols, label="p50")
    ax[1].errorbar(range(6), p50, yerr=[[0] * 6, [a - c for a, c in zip(p95, p50)]], fmt="none", ecolor="black", capsize=3, label="to p95")
    ax[1].set_xticks(range(6))
    ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_ylabel("latency per call (ms)")
    ax[1].set_title("Latency: lower is better (batch 8/32 = whole batch)")
    for i, v in enumerate(p50):
        ax[1].text(i, v + 120, f"{v:,.0f}", ha="center", fontsize=8)
    fig.suptitle("Serving the same model (Qwen2.5-1.5B, 128 tokens/request) on an RTX 4090", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/serving_comparison.png", dpi=130)
    plt.close(fig)


def technique_chart():
    techs = [("bf16", "bf16"), ("grad_ckpt", "gradient\ncheckpointing"), ("grad_accum", "gradient\naccumulation\n(micro 64)"),
             ("adam8bit", "8-bit\noptimiser"), ("combo_bf16_adam8bit", "bf16 +\n8-bit"), ("combo_bf16_ckpt_adam8bit", "bf16 + ckpt\n+ 8-bit")]
    base = json.load(open("results/step4/baseline/raw/bs000512.json"))
    thr, mem = [], []
    for key, _ in techs:
        r = json.load(open(f"results/step4/{key}/raw/bs000512.json"))
        thr.append(r["samples_per_sec"] / base["samples_per_sec"])
        mem.append(r["peak_mem_allocated_gb"] / base["peak_mem_allocated_gb"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = range(len(techs))
    b = ax[0].bar(x, thr, color=["#2ca02c" if v >= 1 else "#d62728" for v in thr])
    ax[0].axhline(1, color="black", lw=1)
    ax[0].set_title("Throughput vs baseline at batch 512 (higher is faster)")
    b2 = ax[1].bar(x, mem, color=["#2ca02c" if v <= 1 else "#d62728" for v in mem])
    ax[1].axhline(1, color="black", lw=1)
    ax[1].set_title("Peak memory vs baseline at batch 512 (lower is better)")
    for a, bars, vals in ((ax[0], b, thr), (ax[1], b2, mem)):
        a.set_xticks(list(x))
        a.set_xticklabels([n for _, n in techs], fontsize=8)
        for r, v in zip(bars, vals):
            a.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}x", ha="center", fontsize=9)
        a.set_ylabel("ratio to fp32 baseline")
    fig.suptitle(f"What each memory technique costs and saves (baseline: {base['samples_per_sec']:.0f} samples/s, "
                 f"{base['peak_mem_allocated_gb']:.1f} GB)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/technique_tradeoffs.png", dpi=130)
    plt.close(fig)


def gpu_hours_chart():
    rows = list(csv.DictReader(open("results/step5/gpu_hours.csv")))
    labels = [r["configuration"].replace(", batch", "\nbatch").replace(" (HW4 baseline)", "\n(baseline)") for r in rows]
    hrs = [float(r["gpu_hours"]) for r in rows]
    secs = [float(r["measured_wallclock_s"]) for r in rows]
    acc = [float(r["test_acc"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    b = ax.bar(range(len(rows)), secs, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#1f77b4"])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("wall-clock for the full fine-tune (seconds)")
    for r, s, h, a in zip(b, secs, hrs, acc):
        ax.text(r.get_x() + r.get_width() / 2, s + 3, f"{s:.0f} s = {h:.4f} GPU-h\ntest acc {a:.2f}%", ha="center", fontsize=8)
    ax.set_ylim(0, max(secs) * 1.25)
    ax.set_title("One full fine-tune (31,930 training rows x 3 epochs) on one RTX 4090")
    fig.tight_layout()
    fig.savefig(f"{OUT}/gpu_hours.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    serving_chart()
    technique_chart()
    gpu_hours_chart()
    print("figures written to", OUT)
