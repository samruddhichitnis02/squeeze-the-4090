"""Generate RUN_LOG.txt and GPU_HOURS_RECORD.md from the per-step run_info.txt files."""
import glob
import os
import re
from datetime import datetime

STEPS = [
    ("step1", "BERT fine-tune on the 5,000-row subset", "bash scripts/step1_run.sh"),
    ("step2", "HF pipeline vs vLLM serving benchmark", "bash scripts/step2_run.sh  (+ scripts/step2_demo.sh)"),
    ("step3", "Batch-size scaling to the memory wall", "bash scripts/step3_run.sh"),
    ("step4", "Memory techniques, one at a time", "bash scripts/step4_run.sh"),
    ("step5", "Full fine-tune GPU-hours check (4 configurations)", "bash scripts/step5_run.sh"),
    ("step6", "Online load test, bf16/fp16/fp8 (main grid)", "bash scripts/step6_run.sh"),
    ("step6_fine", "Online load test, finer rate grid", "RATES='112 144 160 176' SUFFIX=_fine bash scripts/step6_run.sh"),
    ("step6_engine", "Engine-only control (no HTTP), bf16 and fp8", "bash scripts/step6_engine_check.sh"),
    ("step7_diag", "Bottleneck diagnostic: API servers x client processes", "bash scripts/step7_diag.sh"),
    ("step7_diag2", "Bottleneck diagnostic: streaming vs non-streaming", "bash scripts/step7_diag2.sh"),
    ("step7", "Knob sweep, main (7 configurations x 2 workloads; 2 FP8-KV configs failed to start)", "bash scripts/step7_run.sh"),
    ("step7_kv", "Knob sweep, Triton backend (FP8 KV cache and controls)",
     "ONLY='triton_kv_bf16 triton_kv_fp8 triton_w8_kv_bf16 triton_w8_kv_fp8' SUFFIX=_kv bash scripts/step7_run.sh"),
    ("step7_nc", "Knob sweep, prefix cache off (chunked prefill / batched tokens)",
     "ONLY='nocache_nochunk nocache_bt512' WORKLOADS=longprefix SUFFIX=_nc bash scripts/step7_run.sh"),
]

# Events the scripts did not capture. Times marked "approx" come from process elapsed time seen while monitoring.
MANUAL = [
    ("step2", "Smoke tests of vLLM startup (several short runs, each a few minutes at most, three of them failed on "
              "environment problems: missing C compiler, nvcc/header mismatch). GPU time not logged; approx 5 to 10 minutes total."),
    ("step2", "vLLM demo: first attempt failed immediately (missing __main__ guard), second attempt succeeded "
              "(two models, about 2.4 s generation each plus model load). Not timestamped; approx 3 minutes."),
    ("step3", "First sweep attempt stalled at batch 1024 because the Windows driver spilled GPU memory to system RAM "
              "(approx 22 minutes of GPU time at 100% utilization before it was stopped). Its logs were overwritten by the "
              "corrected rerun and its results were discarded."),
    ("step4", "Baseline regression check of the edited train_bert.py: one call hit a 10-minute tool timeout (cause was the output "
              "pipe, GPU use during that time not verified, up to 10 minutes), the rerun finished in about 30 s and reproduced Step 1."),
    ("step4", "bitsandbytes 8-bit optimizer check (scripts/bnb_check.py): a few seconds."),
    ("step5", "First launch failed within 15 s on a command-line flag typo (no GPU work); fixed and rerun."),
    ("step7", "The first knob sweep launch of the two FP8-KV configurations failed at server start (about 1 minute each of GPU "
              "setup, no results). Their server logs are kept in logs/step7/."),
    ("step6", "Server/client smoke test (about 2 minutes) and a first full launch stopped after about 6 minutes because the "
              "rate list did not reach saturation; its results were discarded. The report step failed once for a missing "
              "matplotlib (no GPU work)."),
]


EXTRA_LOGS = {"step6_fine": "logs/step6/run_info_fine.txt", "step6_engine": "logs/step6/run_info_engine_check.txt",
              "step7_diag": "logs/step7/run_info_diag.txt", "step7_diag2": "logs/step7/run_info_diag2.txt",
              "step7_kv": "logs/step7/run_info_kv.txt", "step7_nc": "logs/step7/run_info_nc.txt"}


def parse(step):
    p = EXTRA_LOGS.get(step, f"logs/{step}/run_info.txt")
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    s = re.search(r"run started: (\S+)", txt)
    f = re.search(r"run finished: (\S+)", txt)
    if not (s and f):
        return None
    t0, t1 = datetime.fromisoformat(s.group(1)), datetime.fromisoformat(f.group(1))
    return t0, t1, txt


lines = ["RUN LOG: consolidated record of every scripted GPU run", "=" * 60, ""]
uuid = ""
for p in sorted(glob.glob("logs/step*/gpu_uuid.txt")):
    uuid = open(p).read().strip()
    break
lines += ["Hardware: AMD Ryzen 9 7950X, 128 GB RAM, Windows 11 + WSL2 (Ubuntu 26.04), NVIDIA GeForce RTX 4090 (24 GB)",
          f"GPU UUID capture: {uuid}", "Software: Python 3.12.16, see logs/stepN/pip_freeze.txt for exact package versions", ""]

rec = ["# GPU usage record", "",
       "Single card: NVIDIA GeForce RTX 4090 (own system). Script windows are taken from the 'run started' / 'run finished' "
       "timestamps in `logs/stepN/run_info.txt`.", "",
       "| Step | What ran | Start | Finish | Wall-clock (min) | GPU-hours |", "|---|---|---|---|---|---|"]
total = 0.0
for step, what, cmd in STEPS:
    r = parse(step)
    lines.append(f"[{step}] {what}")
    lines.append(f"  command: {cmd}")
    if not r:
        lines += ["  (no complete run_info.txt)", ""]
        continue
    t0, t1, txt = r
    mins = (t1 - t0).total_seconds() / 60
    total += mins
    lines += [f"  started : {t0.isoformat()}", f"  finished: {t1.isoformat()}", f"  wall-clock: {mins:.1f} min",
              f"  gpu uuid: logs/{step.split('_')[0]}/gpu_uuid*.txt", "  stages:"]
    for l in txt.splitlines():
        if l.startswith("===") or "out of memory" in l or l.startswith("FAILED") or l.startswith("TIMEOUT"):
            lines.append(f"    {l}")
    lines.append("")
    rec.append(f"| {step} | {what} | {t0.strftime('%Y-%m-%d %H:%M')} | {t1.strftime('%H:%M')} | {mins:.1f} | {mins / 60:.3f} |")

rec.append(f"| **Scripted total** | | | | **{total:.1f}** | **{total / 60:.3f}** |")
lines += ["Events not captured by the scripts (approximate, see GPU_HOURS_RECORD.md):"]
rec += ["", "## Not captured by the run scripts (approximate)", ""]
for step, note in MANUAL:
    lines.append(f"  [{step}] {note}")
    rec.append(f"- **{step}**: {note}")
rec += ["", "These unlogged items add roughly 0.5 to 0.75 GPU-hours on top of the scripted total (about 22 minutes for the "
        "stalled sweep, plus up to about 25 minutes of smoke tests, demo attempts and checks). They are estimates, not measurements.", "",
        "## Reservation", "",
        "The GPU is the user's own RTX 4090, so no reservation was needed. See `RESERVATION.md`."]

open("RUN_LOG.txt", "w").write("\n".join(lines) + "\n")
open("GPU_HOURS_RECORD.md", "w").write("\n".join(rec) + "\n")
print("\n".join(rec))
