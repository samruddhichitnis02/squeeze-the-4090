"""Follow-up to kv_fakequant.py: do outlier KEY CHANNELS explain the FP8 KV failure?

Static per-channel scales for keys: for every layer, KV head and head-dim channel, scale = (largest |key| seen on a calibration slice) / 448.
Calibration windows (64..71) are disjoint from the evaluation windows (0..63) of the same WikiText-2 test text.
Values use the same fixed-scale E4M3 as before (that was harmless). Everything simulated in plain PyTorch, no vLLM.
Run in the training env:  python src/kv_fakequant_channel.py
"""
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WINDOW, N_EVAL, N_CAL = 1024, 64, 8
E4M3_MAX = 448.0

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
ids = tok.encode(open("data/wikitext2_test.txt", encoding="utf-8").read())
all_w = [ids[i:i + WINDOW] for i in range(0, len(ids) - WINDOW, WINDOW)]
eval_w, cal_w = all_w[:N_EVAL], all_w[N_EVAL:N_EVAL + N_CAL]


class Recorder(DynamicCache):
    absmax = {}

    def update(self, k, v, layer_idx, cache_kwargs=None):
        m = k.float().abs().amax(dim=(0, 2))  # (heads, head_dim)
        Recorder.absmax[layer_idx] = torch.maximum(Recorder.absmax[layer_idx], m) if layer_idx in Recorder.absmax else m
        return super().update(k, v, layer_idx, cache_kwargs)


with torch.no_grad():
    for w in cal_w:
        model(torch.tensor([w], device="cuda"), past_key_values=Recorder(), use_cache=True)
scales = {l: (m / E4M3_MAX).clamp(min=1e-8) for l, m in Recorder.absmax.items()}


def q8(x, scale=None):
    xf = x.float()
    if scale is None:
        return xf.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float().to(x.dtype)
    s = scale.view(1, -1, 1, scale.shape[-1])  # (1, heads, 1, head_dim)
    return ((xf / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


class QC(DynamicCache):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if self.mode == "k_channel_v_fixed":
            k, v = q8(k, scales[layer_idx]), q8(v)
        elif self.mode == "k_channel_only":
            k = q8(k, scales[layer_idx])
        return super().update(k, v, layer_idx, cache_kwargs)


@torch.no_grad()
def ppl(make):
    nll, n = 0.0, 0
    for w in eval_w:
        x = torch.tensor([w], device="cuda")
        lg = model(x, past_key_values=make(), use_cache=True).logits.float()
        nll -= torch.log_softmax(lg[0, :-1], -1).gather(1, x[0, 1:, None]).sum().item()
        n += x.shape[1] - 1
    return math.exp(nll / n)


res = []
for name, make in (("no quantisation (reference)", lambda: DynamicCache()),
                   ("keys per-channel static E4M3 scale, values fixed-scale E4M3", lambda: QC("k_channel_v_fixed")),
                   ("keys per-channel static E4M3 scale only", lambda: QC("k_channel_only"))):
    p = ppl(make)
    res.append(dict(variant=name, perplexity=p))
    print(f"{p:8.3f}  {name}", flush=True)
spread = {l: float((s * E4M3_MAX).max() / (s * E4M3_MAX).median()) for l, s in scales.items()}
print("calibrated key channel range spread (max / median per-channel absmax), layers 0,1,2,10,27:",
      {l: round(spread[l], 1) for l in (0, 1, 2, 10, 27) if l in spread})
os.makedirs("results/step8", exist_ok=True)
json.dump(dict(model=MODEL, eval_windows=N_EVAL, cal_windows=N_CAL, results=res, channel_spread_by_layer=spread),
          open("results/step8/kv_fakequant_channel.json", "w"), indent=1)
