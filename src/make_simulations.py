"""Animated explainers for the README. Every parameter comes from measured data in results/.

1. continuous_batching.gif : static vs continuous batching on identical arrivals (illustrative scale: 16 slots).
2. latency_vs_load.gif/.png : simulated latency vs offered load (bf16 step-time model) against the measured Step 6 points.
3. decode_roofline.gif      : why adding requests to a decode batch is nearly free at first (bandwidth vs compute).
4. memory_wall.gif          : peak GPU memory as batch size doubles until the out-of-memory error (Step 3, measured).

Step-time model (fitted in fit_step_model.py on Step 6 bf16 data): step_ms = A + C * requests_in_flight.
"""
import glob
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle

from fit_step_model import fit, load

OUT = "docs/img"
os.makedirs(OUT, exist_ok=True)
A, C, _, _, _ = fit("bf16")  # ms, ms per request in flight
OVERHEAD_MS = 10.0           # HTTP + tokenisation + prefill for short prompts (measured TTFT 13-15 ms at low load minus one step)
BW_STEP_MS = 3.06            # weights read once at 1,008 GB/s (3.09 GB)
COMPUTE_MS_PER_REQ = 3.09e9 / 165.6e12 * 1000   # decode matmul FLOPs per request at 100% of the bf16 tensor peak
ENGINE_ONLY_STEP_MS = 1138.6 / 128              # Step 6 engine-only, 256 sequences: p50 call 1138.6 ms / 128 tokens
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
FPS = 10


# ---------------------------------------------------------------- core simulator (iteration-level, same logic as the HTML)
def simulate(rate, cap=256, out_len=128, duration=25.0, seed=2122, a=A, c=C, overhead_ms=OVERHEAD_MS):
    rng = random.Random(seed)
    t, nxt = 0.0, rng.expovariate(rate)
    waiting, active, lat = [], [], []
    while t < duration:
        while nxt <= t:
            waiting.append(nxt)
            nxt += rng.expovariate(rate)
        while len(active) < cap and waiting:
            active.append([out_len, waiting.pop(0)])
        if not active:
            t = min(nxt, duration)
            continue
        t += (a + c * len(active)) / 1000
        for r in active:
            r[0] -= 1
        for r in [r for r in active if r[0] <= 0]:
            lat.append((t - r[1]) * 1000 + overhead_ms)
        active = [r for r in active if r[0] > 0]
    return np.array(lat), len(lat) / duration


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


# ---------------------------------------------------------------- 1. continuous vs static batching
def trace(policy, arrivals, lengths, cap, frame_steps, t_end):
    """Return frames: list of dict(t, slots[(remaining,total)|None|'idle'], queue, done, lat_sum)."""
    t, i_arr, waiting, slots, done, lat_sum = 0.0, 0, [], [None] * cap, 0, 0.0
    frames, steps = [], 0
    static_batch_open = True
    while t < t_end:
        while i_arr < len(arrivals) and arrivals[i_arr] <= t:
            waiting.append((arrivals[i_arr], lengths[i_arr]))
            i_arr += 1
        if policy == "continuous":
            for s in range(cap):
                if slots[s] is None and waiting:
                    arr, ln = waiting.pop(0)
                    slots[s] = [ln, ln, arr]
        else:  # static: refill only when every slot is free
            if all(s is None or s == "idle" for s in slots) and waiting:
                slots = [None] * cap
                for s in range(cap):
                    if waiting:
                        arr, ln = waiting.pop(0)
                        slots[s] = [ln, ln, arr]
        live = [s for s in slots if isinstance(s, list)]
        if steps % frame_steps == 0:
            frames.append(dict(t=t, slots=[(list(s) if isinstance(s, list) else s) for s in slots],
                               queue=len(waiting), done=done, lat_sum=lat_sum))
        if not live:
            t += 0.002
            steps += 1
            continue
        t += (A + C * len(live)) / 1000
        steps += 1
        for k, s in enumerate(slots):
            if isinstance(s, list):
                s[0] -= 1
                if s[0] <= 0:
                    done += 1
                    lat_sum += (t - s[2]) * 1000
                    slots[k] = None if policy == "continuous" else "idle"
    return frames


def make_batching_gif():
    cap, t_end = 16, 3.0
    rng = random.Random(2122)
    rate = 55.0
    arrivals, tt = [], 0.0
    while tt < t_end:
        tt += rng.expovariate(rate)
        arrivals.append(tt)
    lengths = [rng.randint(16, 96) for _ in arrivals]     # varied output lengths: this is what static batching suffers from
    fs = {p: trace(p, arrivals, lengths, cap, 6, t_end) for p in ("static", "continuous")}
    n = min(len(fs["static"]), len(fs["continuous"]))
    fig = plt.figure(figsize=(10, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.6])
    axes = {"static": fig.add_subplot(gs[0, 0]), "continuous": fig.add_subplot(gs[0, 1])}
    axc = fig.add_subplot(gs[1, :])
    cmap = plt.cm.viridis
    titles = {"static": "Static batching: wait for the whole batch to finish",
              "continuous": "Continuous batching: refill a slot the moment it frees"}

    def draw(k):
        for pol, ax in axes.items():
            ax.clear()
            f = fs[pol][min(k, len(fs[pol]) - 1)]
            for s, cell in enumerate(f["slots"]):
                x, y = s % 4, 3 - s // 4
                if isinstance(cell, list):
                    prog = 1 - cell[0] / cell[1]
                    ax.add_patch(Rectangle((x, y), 0.92, 0.92, color=cmap(0.15 + 0.8 * prog)))
                elif cell == "idle":
                    ax.add_patch(Rectangle((x, y), 0.92, 0.92, color="#d9d9d9", hatch="//", ec="white"))
                else:
                    ax.add_patch(Rectangle((x, y), 0.92, 0.92, color="#f2f2f2"))
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 4.9)
            ax.axis("off")
            ax.set_title(titles[pol], fontsize=9)
            ax.text(0, 4.15, f"waiting in queue: {f['queue']}   finished: {f['done']}", fontsize=9)
        axc.clear()
        for pol, col in (("static", "#d95f02"), ("continuous", "#1b9e77")):
            ts = [f["t"] for f in fs[pol][:k + 1]]
            ds = [f["done"] for f in fs[pol][:k + 1]]
            axc.plot(ts, ds, color=col, label=pol)
        axc.set_xlim(0, t_end)
        axc.set_ylim(0, max(fs["continuous"][-1]["done"], fs["static"][-1]["done"]) * 1.1)
        axc.set_xlabel("simulated seconds")
        axc.set_ylabel("requests finished")
        axc.legend(loc="upper left", fontsize=8)
        fig.suptitle("Same arrivals (55 req/s), same lengths (16 to 96 tokens), 16 slots.\n"
                     "Colour = progress through the answer, grey = slot idle", fontsize=9)

    ani = animation.FuncAnimation(fig, draw, frames=n, interval=100)
    ani.save(f"{OUT}/continuous_batching.gif", writer=animation.PillowWriter(fps=FPS), dpi=70)
    plt.close(fig)
    last = {p: fs[p][-1] for p in fs}
    return {p: dict(finished=last[p]["done"], mean_latency_ms=last[p]["lat_sum"] / max(last[p]["done"], 1)) for p in last}


# ---------------------------------------------------------------- 2. latency vs load
def make_latency_gif():
    measured = load("bf16")
    rates = np.array([1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240])
    p50, p99, tput = [], [], []
    for r in rates:
        lat, thr = simulate(float(r), cap=256, duration=30.0)
        p50.append(pct(lat, 50))
        p99.append(pct(lat, 99))
        tput.append(thr)
    p50, p99, tput = np.array(p50), np.array(p99), np.array(tput)
    # validation: sim p50 vs measured p50 where the API stack is not the limit (offered <= 128)
    val = []
    for m in measured:
        if m["offered_rate"] <= 128:
            lat, _ = simulate(m["offered_rate"], duration=30.0)
            val.append((m["offered_rate"], m["e2e_ms"]["p50"], pct(lat, 50)))
    err = [abs(s - m) / m * 100 for _, m, s in val]
    mx = [m["offered_rate"] for m in measured]
    mp50 = [m["e2e_ms"]["p50"] for m in measured]
    mp99 = [m["e2e_ms"]["p99"] for m in measured]

    def frame_fig(k):
        ax.clear()
        ax.plot(rates[:k + 1], p50[:k + 1], "-", color="#1b9e77", lw=2, label="simulated p50")
        ax.plot(rates[:k + 1], p99[:k + 1], "-", color="#7570b3", lw=2, label="simulated p99")
        shown = [i for i, x in enumerate(mx) if x <= rates[k]]
        ax.plot([mx[i] for i in shown], [mp50[i] for i in shown], "o", color="#1b9e77", mfc="white", label="measured p50 (online)")
        ax.plot([mx[i] for i in shown], [mp99[i] for i in shown], "s", color="#7570b3", mfc="white", label="measured p99 (online)")
        ax.axvline(140, color="gray", ls=":", lw=1)
        ax.text(141, 20000, "online stack plateau\n(~140 req/s measured)", fontsize=8, color="gray", va="top")
        ax.axvline(224.6, color="#d95f02", ls=":", lw=1)
        ax.text(225, 640, "engine-only\ncapacity\n(224.6 req/s measured)", fontsize=8, color="#d95f02", va="bottom")
        ax.set_yscale("log")
        ax.set_xlim(0, 245)
        ax.set_ylim(500, 30000)
        ax.set_xlabel("offered load (requests / second)")
        ax.set_ylabel("end-to-end latency (ms, log scale)")
        ax.set_title("Latency stays flat, then explodes near capacity (queueing)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3, which="both")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    frames = list(range(len(rates))) + [len(rates) - 1] * 8
    ani = animation.FuncAnimation(fig, lambda k: frame_fig(k), frames=frames, interval=150)
    ani.save(f"{OUT}/latency_vs_load.gif", writer=animation.PillowWriter(fps=6), dpi=75)
    frame_fig(len(rates) - 1)
    fig.tight_layout()
    fig.savefig(f"{OUT}/latency_vs_load.png", dpi=130)
    plt.close(fig)
    return dict(model=dict(a_ms=A, c_ms_per_request=C, overhead_ms=OVERHEAD_MS),
                validation=[dict(rate=r, measured_p50=m, simulated_p50=s) for r, m, s in val],
                mean_abs_err_pct=float(np.mean(err)), max_abs_err_pct=float(np.max(err)),
                sim_capacity_req_s=float(tput[-1]),
                sim_tokens_per_sec_at_256=float(256 / ((A + C * 256) / 1000)))


# ---------------------------------------------------------------- 3. decode roofline sweep
def make_roofline_gif():
    L = np.arange(1, 257)
    mem = np.full_like(L, BW_STEP_MS, dtype=float)
    comp = COMPUTE_MS_PER_REQ * L
    fitted = A + C * L
    measured = load("bf16")
    ml = [m["achieved_req_per_sec"] * m["e2e_ms"]["mean"] / 1000 for m in measured if m["offered_rate"] <= 128]
    mt = [m["tpot_ms"]["p50"] for m in measured if m["offered_rate"] <= 128]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    def draw(k):
        ax.clear()
        upto = L[:k + 1]
        ax.plot(L, mem, "--", color="#1f77b4", label=f"time to read the weights once ({BW_STEP_MS} ms at 1,008 GB/s)")
        ax.plot(L, comp, "--", color="#d62728", label="time for the matrix maths at 100% of tensor peak")
        ax.plot(upto, fitted[:k + 1], "-", color="black", lw=2, label="measured step time (fit to Step 6)")
        ax.plot(ml, mt, "o", color="black", mfc="white", label="measured points (online load test)")
        ax.plot([256], [ENGINE_ONLY_STEP_MS], "*", color="#d95f02", ms=14, label="measured, engine-only, 256 seqs")
        ax.axvline(upto[-1], color="gray", lw=0.8)
        ax.text(min(upto[-1] + 3, 190), 11.2, f"{int(upto[-1])} requests in flight\nstep = {fitted[k]:.2f} ms", fontsize=9, va="top")
        ax.set_xlim(0, 262)
        ax.set_ylim(0, 12)
        ax.set_xlabel("requests being decoded together (in flight)")
        ax.set_ylabel("time per decode step (ms)")
        ax.set_title("Decode: the first requests ride along almost for free")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)

    frames = list(range(0, 256, 6)) + [255] * 10
    ani = animation.FuncAnimation(fig, draw, frames=frames, interval=120)
    ani.save(f"{OUT}/decode_roofline.gif", writer=animation.PillowWriter(fps=8), dpi=75)
    draw(255)
    fig.tight_layout()
    fig.savefig(f"{OUT}/decode_roofline.png", dpi=130)
    plt.close(fig)
    return dict(a=A, c=C, compute_ms_per_req=COMPUTE_MS_PER_REQ, bw_ms=BW_STEP_MS, engine_step_ms=ENGINE_ONLY_STEP_MS,
                fit_at_256=A + C * 256)


# ---------------------------------------------------------------- 4. memory wall (measured, Step 3)
def make_memory_gif():
    runs = sorted((json.load(open(p)) for p in glob.glob("results/step3/raw/*.json")), key=lambda r: r["batch_size"])
    cap = runs[0]["usable_gb"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    def draw(k):
        ax.clear()
        for i, r in enumerate(runs[:k + 1]):
            if r["status"] == "ok":
                ax.barh(i, r["peak_mem_allocated_gb"], color=plt.cm.YlOrRd(0.2 + 0.6 * r["peak_mem_allocated_gb"] / cap))
                ax.text(r["peak_mem_allocated_gb"] + 0.2, i, f"{r['peak_mem_allocated_gb']:.1f} GB  |  {r['samples_per_sec']:.0f} samples/s",
                        va="center", fontsize=8)
            else:
                ax.barh(i, cap, color="#b2182b", alpha=0.85)
                ax.text(cap / 2, i, "OUT OF MEMORY", ha="center", va="center", color="white", fontweight="bold")
        ax.axvline(cap, color="black", ls=":")
        ax.text(cap, len(runs) - 0.35, f"usable VRAM {cap:.1f} GB", ha="right", fontsize=8)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([f"batch {r['batch_size']}" for r in runs])
        ax.invert_yaxis()
        ax.set_xlim(0, cap + 4)
        ax.set_xlabel("peak GPU memory during training (GB)")
        ax.set_title("Double the batch, the memory wall gets closer (RTX 4090, BERT fine-tune, fp32)")

    frames = list(range(len(runs))) + [len(runs) - 1] * 6
    ani = animation.FuncAnimation(fig, draw, frames=frames, interval=700)
    ani.save(f"{OUT}/memory_wall.gif", writer=animation.PillowWriter(fps=2), dpi=75)
    plt.close(fig)
    return dict(cap=cap, sizes=[r["batch_size"] for r in runs])


if __name__ == "__main__":
    summary = {}
    summary["batching"] = make_batching_gif()
    summary["latency"] = make_latency_gif()
    summary["roofline"] = make_roofline_gif()
    summary["memory"] = make_memory_gif()
    json.dump(summary, open(f"{OUT}/simulation_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2)[:3000])
