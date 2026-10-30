"""Fit decode step time vs requests in flight from the Step 6 load test (bf16), for the README simulations.

Little's law: requests in flight L = achieved_req_per_sec x mean e2e latency. TPOT p50 is the time per decode step seen by a stream.
Fit TPOT = a + c * L on rates below the API-stack plateau, then check the prediction at 256 sequences against the
engine-only measurement (28,754 tok/s), which is NOT used in the fit.
"""
import glob
import json

import numpy as np


def load(label):
    rates = []
    for p in sorted(glob.glob(f"results/step6/raw/{label}*.json")):
        rates += json.load(open(p))["rates"]
    seen, out = set(), []
    for r in sorted(rates, key=lambda r: r["offered_rate"]):
        if r["offered_rate"] not in seen:
            seen.add(r["offered_rate"])
            out.append(r)
    return out


def fit(label, max_rate=128):
    rs = [r for r in load(label) if r["offered_rate"] <= max_rate]
    L = np.array([r["achieved_req_per_sec"] * r["e2e_ms"]["mean"] / 1000 for r in rs])
    t = np.array([r["tpot_ms"]["p50"] for r in rs])
    c, a = np.polyfit(L, t, 1)
    resid = t - (a + c * L)
    return a, c, L, t, resid


if __name__ == "__main__":
    for label in ("bf16", "fp8"):
        a, c, L, t, resid = fit(label)
        print(f"{label}: TPOT(ms) = {a:.2f} + {c:.4f} x L   (points {len(L)}, max |resid| {abs(resid).max():.2f} ms)")
        for Li, ti in zip(L, t):
            print(f"   L={Li:6.1f}  TPOT={ti:5.2f}")
        step256 = a + c * 256
        print(f"   predicted step at L=256: {step256:.2f} ms -> {256 / (step256 / 1000) :.0f} tok/s "
              f"(engine-only measured: {'28754' if label == 'bf16' else '28730'})")
