"""Largest key/value magnitudes per layer for Qwen2.5-1.5B-Instruct, to see whether an FP8 KV cache can clip.

FP8 E4M3 saturates at 448; E5M2 at 57,344. Keys are measured AFTER rotary embedding (what the cache stores).
Run in the training environment: python src/kv_range.py
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
text = open("data/wikitext2_test.txt", encoding="utf-8").read()[:60000]
ids = tok(text, return_tensors="pt").input_ids[:, :1024].cuda()
with torch.no_grad():
    out = model(ids, use_cache=True)
pkv = out.past_key_values
rows = []
for i in range(len(pkv)):
    k, v = pkv[i][0].float(), pkv[i][1].float()
    rows.append(dict(layer=i, k_absmax=float(k.abs().max()), v_absmax=float(v.abs().max()),
                     k_p999=float(torch.quantile(k.abs().flatten()[:2_000_000], 0.999)),
                     v_p999=float(torch.quantile(v.abs().flatten()[:2_000_000], 0.999))))
summary = dict(
    model=MODEL, tokens=int(ids.shape[1]),
    k_absmax=max(r["k_absmax"] for r in rows), v_absmax=max(r["v_absmax"] for r in rows),
    layers_k_over_448=[r["layer"] for r in rows if r["k_absmax"] > 448],
    layers_v_over_448=[r["layer"] for r in rows if r["v_absmax"] > 448],
    e4m3_max=448, e5m2_max=57344, per_layer=rows)
os.makedirs("results/step8", exist_ok=True)
json.dump(summary, open("results/step8/kv_range.json", "w"), indent=1)
print(f"max |K| = {summary['k_absmax']:.1f}   max |V| = {summary['v_absmax']:.1f}   (E4M3 limit 448, E5M2 limit 57,344)")
print("layers with |K| > 448:", summary["layers_k_over_448"])
print("layers with |V| > 448:", summary["layers_v_over_448"])
top = sorted(rows, key=lambda r: -r["k_absmax"])[:4]
for r in top:
    print(f"  layer {r['layer']:2d}: |K|max {r['k_absmax']:8.1f} (99.9th pct {r['k_p999']:6.1f})   |V|max {r['v_absmax']:6.1f}")
