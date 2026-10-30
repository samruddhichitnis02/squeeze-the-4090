"""Test docs/simulator.html logic in QuickJS (no browser available): lookups vs saved results, simulation sanity,
and a smoke run of every UI render path against a stubbed DOM. Needs: pip install quickjs.
This does NOT test real rendering (canvas drawing, layout, CSS) - open the page in a browser for that."""
import json
import re
import sys

import quickjs

html = open("docs/simulator.html", encoding="utf-8").read()
script = re.search(r"<script>(.*)</script>", html, re.S).group(1)
data = json.loads(re.search(r"const DATA = (\{.*?\});\n", script, re.S).group(1))
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


# ---- 1. pure logic section
logic = script[script.index("const DATA"):script.index("// ==UI==")]
ctx = quickjs.Context()
ctx.eval(logic)
J = lambda expr: json.loads(ctx.eval(f"JSON.stringify({expr})"))

b512 = json.load(open("results/step4/baseline/raw/bs000512.json"))
r = J("lookupTrain('baseline', 512)")
check("train lookup matches saved baseline@512", r["state"] == "ok" and abs(r["mem"] - b512["peak_mem_allocated_gb"]) < 1e-9
      and abs(r["sps"] - b512["samples_per_sec"]) < 1e-9, f"{r}")
check("baseline@1024 is a measured OOM", J("lookupTrain('baseline', 1024)")["state"] == "oom")
check("baseline@4096 is inferred OOM", J("lookupTrain('baseline', 4096)")["state"] == "oom-inferred")
check("grad_ckpt@2048 runs", J("lookupTrain('grad_ckpt', 2048)")["state"] == "ok")
check("grad_ckpt@4096 is a measured OOM", J("lookupTrain('grad_ckpt', 4096)")["state"] == "oom")
check("grad_accum@8 is untested (sweep began at 64)", J("lookupTrain('grad_accum', 8)")["state"] == "untested")
ft = J("fullFineTune(%r)" % b512["samples_per_sec"])
check("full fine-tune estimate = 95,790 / sps", abs(ft["seconds"] - 95790 / b512["samples_per_sec"]) < 1e-6, f"{ft['seconds']:.1f}s")

s6 = json.load(open("results/step6/raw/bf16.json"))["rates"]
m128 = next(x for x in s6 if x["offered_rate"] == 128)
sv = J("lookupServe('bf16', 128)")
check("serve lookup matches saved bf16@128", abs(sv["e2e_p50"] - m128["e2e_ms"]["p50"]) < 1e-9 and abs(sv["tok_s"] - m128["output_tokens_per_sec"]) < 1e-9)
check("serve rates list is sorted and complete", J("RATES") == sorted(J("RATES")) and len(J("RATES")) >= 13, str(J("RATES")))
check("unknown rate returns null", ctx.eval("lookupServe('bf16', 7777) === null"))

# simulation sanity (JS RNG differs from Python's, so compare statistically with tolerance)
sim = J("(() => { const r = simulate(96, 256, 128, 30, 2122, 0); return {p50:r.p50, p99:r.p99, rps:r.reqPerSec, tps:r.tokPerSec}; })()")
meas96 = next(x for x in s6 if x["offered_rate"] == 96)["e2e_ms"]["p50"]
check("sim p50 at 96 req/s within 15% of measured", abs(sim["p50"] - meas96) / meas96 < 0.15, f"sim {sim['p50']:.0f} vs measured {meas96:.0f} ms")
check("sim serves ~all 96 req/s", abs(sim["rps"] - 96) / 96 < 0.06, f"{sim['rps']:.1f}")
cap32 = J("(() => { const r = simulate(160, 32, 128, 30, 2122, 0); return r.tokPerSec; })()")
a, c = data["model"]["a_ms"], data["model"]["c_ms"]
expect32 = 32 / ((a + c * 32) / 1000)
check("cap=32 saturates at 32/step_time tokens/s", abs(cap32 - expect32) / expect32 < 0.05, f"{cap32:.0f} vs {expect32:.0f}")
low = J("(() => { const r = simulate(4, 256, 128, 30, 2122, 0); return r.p50; })()")
check("low-load latency ~ 128 steps x 4.8 ms + overhead", 600 < low < 700, f"{low:.0f} ms")
over = J("(() => { const r = simulate(400, 256, 128, 20, 2122, 0); return {p50:r.p50, rps:r.reqPerSec}; })()")
check("overload: served rate falls below offered and latency grows", over["rps"] < 300 and over["p50"] > 1500, f"{over}")

# ---- 2. UI smoke run against a stubbed DOM
stub = r"""
var __els = {};
function __mk(id){ return {id:id, value:'', textContent:'', innerHTML:'', className:'', style:{}, max:'', _l:{},
  addEventListener:function(t,f){ this._l[t]=f; },
  getContext:function(){ return new Proxy({}, {get:function(t,p){ if(p==='measureText') return function(){return {width:10};}; return function(){}; },
                                                   set:function(){ return true; }}); },
  width:1000, height:360}; }
var document = {getElementById:function(id){ if(!__els[id]) __els[id]=__mk(id); return __els[id]; }};
var __timers = [];
function setInterval(f){ __timers.push(f); return __timers.length; } function clearInterval(){}
// Browser behaviour the stub must mimic: sliders/selects start at the HTML defaults, and a <select> takes its first option.
var __defaults = {prec:'bf16', rate:'6', bs:'5', srate:'96', scap:'256', sout:'128'};
Object.keys(__defaults).forEach(function(k){ document.getElementById(k).value = __defaults[k]; });
(function(){ var el = document.getElementById('cfg'); var html = '';
  Object.defineProperty(el, 'innerHTML', {get:function(){ return html; },
    set:function(h){ html = h; var m = /value="([^"]+)"/.exec(h); if (m) el.value = m[1]; }}); })();
"""
ui_ctx = quickjs.Context()
ui_ctx.eval(stub)
ui_ctx.eval(script)
ui_ctx.eval("__els['prec'].value = 'bf16'; __els['rate'].value = '0';")
n_train = n_serve = 0
errors = []
for cfg in data["train"]:
    for i in range(10):
        ui_ctx.eval(f"__els['cfg'].value = {json.dumps(cfg)}; __els['bs'].value = '{i}';")
        try:
            ui_ctx.eval("renderTrain()")
            out = ui_ctx.eval("__els['s1'].innerHTML + __els['n1'].textContent")
            if "NaN" in out or "undefined" in out:
                errors.append(("train", cfg, i, out[:120]))
            n_train += 1
        except Exception as e:  # noqa: BLE001
            errors.append(("train", cfg, i, repr(e)))
for prec in data["serve"]:
    for i in range(len(data["serve"][prec])):
        ui_ctx.eval(f"__els['prec'].value = {json.dumps(prec)}; __els['rate'].value = '{i}';")
        try:
            ui_ctx.eval("renderServe()")
            out = ui_ctx.eval("__els['s2'].innerHTML + __els['n2'].textContent")
            if "NaN" in out or "undefined" in out:
                errors.append(("serve", prec, i, out[:120]))
            n_serve += 1
        except Exception as e:  # noqa: BLE001
            errors.append(("serve", prec, i, repr(e)))
n_sim = 0
for rate, cap, out_len in [(1, 4, 16), (96, 256, 128), (260, 256, 256), (200, 32, 128), (50, 64, 128)]:
    ui_ctx.eval(f"__els['srate'].value='{rate}'; __els['scap'].value='{cap}'; __els['sout'].value='{out_len}';")
    try:
        ui_ctx.eval("runSim()")
        ui_ctx.eval("for (let k = 0; k < 600 && __timers.length; k++) { __timers[__timers.length-1](); }")
        out = ui_ctx.eval("__els['s3'].innerHTML + __els['n3'].textContent")
        if "NaN" in out or "undefined" in out:
            errors.append(("sim", rate, cap, out[:120]))
        n_sim += 1
    except Exception as e:  # noqa: BLE001
        errors.append(("sim", rate, cap, repr(e)))
check(f"UI smoke: {n_train} training views, {n_serve} serving views, {n_sim} simulations, no exceptions/NaN", not errors, str(errors[:3]))

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
