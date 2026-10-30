"""Does an FP8 KV cache hurt Qwen2.5-1.5B-Instruct in principle? Simulate it in plain PyTorch (no vLLM) and measure perplexity.

A DynamicCache subclass quantises the new keys/values (after rotary embedding, exactly what a cache stores) to FP8 and back
before storing them, so every attention read sees quantised K/V. Variants: which tensors (K, V, both), the format
(E4M3, E5M2) and the scale (fixed 1.0 like vLLM's `fp8`, or per-token-per-head dynamic like `fp8_per_token_head`).
Same WikiText-2 windows for every variant. Run in the training env:  python src/kv_fakequant.py
"""
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WINDOW, N_WINDOWS = 1024, 64
E4M3_MAX, E5M2_MAX = 448.0, 57344.0

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
ids = tok.encode(open("data/wikitext2_test.txt", encoding="utf-8").read())
windows = [ids[i:i + WINDOW] for i in range(0, len(ids) - WINDOW, WINDOW)][:N_WINDOWS]


def fq(x, fmt, scale_mode):
    """Quantise-dequantise x (batch, heads, tokens, head_dim) through FP8."""
    dt = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    mx = E4M3_MAX if fmt == "e4m3" else E5M2_MAX
    xf = x.float()
    if scale_mode == "fixed":
        q = xf.clamp(-mx, mx).to(dt).float()
    else:  # per token per head: scale so the largest element maps to the format maximum
        s = (xf.abs().amax(dim=-1, keepdim=True) / mx).clamp(min=1e-12)
        q = (xf / s).clamp(-mx, mx).to(dt).float() * s
    return q.to(x.dtype)


class QuantCache(DynamicCache):
    def __init__(self, which, fmt, scale_mode):
        super().__init__()
        self.which, self.fmt, self.scale_mode = which, fmt, scale_mode

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if "k" in self.which:
            key_states = fq(key_states, self.fmt, self.scale_mode)
        if "v" in self.which:
            value_states = fq(value_states, self.fmt, self.scale_mode)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


@torch.no_grad()
def perplexity(make_cache):
    nll, count = 0.0, 0
    for w in windows:
        x = torch.tensor([w], device="cuda")
        cache = make_cache()
        logits = model(x, past_key_values=cache, use_cache=True).logits.float()
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        nll -= lp.gather(1, x[0, 1:].unsqueeze(1)).sum().item()
        count += x.shape[1] - 1
    return math.exp(nll / count)


variants = [("no quantisation (reference)", None)]
for scale_mode in ("fixed", "per_token_head"):
    for fmt in ("e4m3", "e5m2"):
        for which in ("kv", "k", "v"):
            variants.append((f"{fmt} {scale_mode} scale, {which.upper()} quantised", (which, fmt, scale_mode)))

results = []
for name, spec in variants:
    ppl = perplexity((lambda: DynamicCache()) if spec is None else (lambda s=spec: QuantCache(*s)))
    results.append(dict(variant=name, perplexity=ppl))
    print(f"{ppl:8.3f}  {name}", flush=True)

os.makedirs("results/step8", exist_ok=True)
json.dump(dict(model=MODEL, windows=N_WINDOWS, window=WINDOW, tokens_scored=N_WINDOWS * (WINDOW - 1), results=results),
          open("results/step8/kv_fakequant.json", "w"), indent=1)
