"""Build docs/simulator.html: a self-contained interactive explorer with the measured data embedded.

Tab 1 looks up MEASURED training memory/speed for a configuration and batch size (Steps 3 and 4).
Tab 2 looks up MEASURED serving latency/throughput for a precision and offered load (Step 6).
Tab 3 runs a small iteration-level serving SIMULATION (model fitted on Step 6 bf16 data, see fit_step_model.py).
"""
import glob
import json
import os

from fit_step_model import fit

A, C, _, _, _ = fit("bf16")

TRAIN = [
    ("baseline", "Baseline (fp32)"),
    ("bf16", "Mixed precision (bf16)"),
    ("grad_ckpt", "Gradient checkpointing"),
    ("grad_accum", "Gradient accumulation (micro-batch 64)"),
    ("adam8bit", "8-bit optimiser"),
    ("combo_bf16_adam8bit", "bf16 + 8-bit optimiser"),
    ("combo_bf16_ckpt_adam8bit", "bf16 + checkpointing + 8-bit optimiser"),
]


def train_points(key):
    pts = []
    for p in sorted(glob.glob(f"results/step4/{key}/raw/*.json")):
        r = json.load(open(p))
        if r["status"] == "ok":
            pts.append(dict(batch=r["batch_size"], status="ok", sps=r["samples_per_sec"], mem=r["peak_mem_allocated_gb"]))
        else:
            pts.append(dict(batch=r["batch_size"], status="oom"))
    return sorted(pts, key=lambda x: x["batch"])


def merged_rates(label, base_dir="results/step6/raw"):
    rates = []
    for p in sorted(glob.glob(f"{base_dir}/{label}*.json")):
        rates += json.load(open(p))["rates"]
    seen, out = set(), []
    for r in sorted(rates, key=lambda r: r["offered_rate"]):
        if r["offered_rate"] in seen:
            continue
        seen.add(r["offered_rate"])
        out.append(dict(offered=r["offered_rate"], achieved=r["achieved_req_per_sec"], tok_s=r["output_tokens_per_sec"],
                        e2e_p50=r["e2e_ms"]["p50"], e2e_p99=r["e2e_ms"]["p99"], ttft_p50=r["ttft_ms"]["p50"],
                        ttft_p99=r["ttft_ms"]["p99"], tpot_p50=r["tpot_ms"]["p50"], goodput=r["goodput_req_per_sec"]))
    return out


def knob_peak(label):
    r = json.load(open(f"results/step7/raw/{label}__short.json"))["rates"]
    return max(x["output_tokens_per_sec"] for x in r)


cap_gb = json.load(open("results/step3/raw/bs000008.json"))["usable_gb"]
data = dict(
    cap_gb=cap_gb,
    full_finetune_samples=31930 * 3,
    train={k: dict(label=lbl, points=train_points(k)) for k, lbl in TRAIN},
    serve={k: merged_rates(k) for k in ("bf16", "fp16", "fp8")},
    seq_cap_measured={"32": knob_peak("seqs_32"), "64": knob_peak("seqs_64"), "256": knob_peak("base")},
    engine_only_tok_s=28753.6,
    model=dict(a_ms=A, c_ms=C, overhead_ms=10.0),
)

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inference optimisation lab: interactive simulator</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f5f6f8;color:#1c1e21}
 header{background:#1f2937;color:#fff;padding:14px 22px}
 header h1{margin:0;font-size:20px} header p{margin:4px 0 0;font-size:13px;color:#cbd5e1}
 nav{display:flex;gap:4px;padding:10px 22px 0;background:#e5e7eb;flex-wrap:wrap}
 nav button{border:0;padding:10px 16px;background:#d1d5db;cursor:pointer;border-radius:8px 8px 0 0;font-size:14px}
 nav button.on{background:#fff;font-weight:600}
 main{padding:16px 22px;max-width:1100px;margin:0 auto}
 .card{background:#fff;border-radius:10px;padding:14px 16px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 .row{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:8px 0}
 label{font-size:13px;color:#374151} select,input[type=range]{margin-left:8px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:10px}
 .stat{background:#f3f4f6;border-radius:8px;padding:8px 10px} .stat b{display:block;font-size:19px}
 .stat span{font-size:12px;color:#6b7280}
 .ok{color:#047857} .bad{color:#b91c1c} .warn{color:#b45309}
 .note{font-size:12.5px;color:#4b5563;margin-top:8px;line-height:1.45}
 canvas{max-width:100%;background:#fff;border:1px solid #e5e7eb;border-radius:8px}
 button.act{padding:7px 14px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer;font-size:14px}
 .tag{display:inline-block;font-size:11px;padding:2px 7px;border-radius:10px;background:#dbeafe;color:#1e40af;margin-left:6px}
 .tag.sim{background:#fef3c7;color:#92400e}
</style></head><body>
<header><h1>Inference optimisation lab: interactive simulator</h1>
<p>RTX 4090, BERT fine-tuning and Qwen2.5-1.5B serving. Tabs 1 and 2 show <b>measured</b> numbers from this repository; tab 3 is a <b>simulation</b>.</p></header>
<nav><button id="t1" class="on">1. Training memory wall <span class="tag">measured</span></button>
<button id="t2">2. Serving under load <span class="tag">measured</span></button>
<button id="t3">3. Batching simulator <span class="tag sim">simulation</span></button></nav>
<main>
<section id="p1">
 <div class="card"><div class="row">
  <label>Technique <select id="cfg"></select></label>
  <label>Batch size <input id="bs" type="range" min="0" max="9" step="1" value="5"><b id="bsv"></b></label></div>
  <div class="grid" id="s1"></div><div class="note" id="n1"></div></div>
 <div class="card"><canvas id="c1" width="1000" height="360"></canvas>
  <div class="note">Each line is one technique, thick = selected. Red x = a measured out-of-memory failure. Dotted line = usable VRAM cap
  (PyTorch is capped at 95% of the memory free at start, so overflow is a real error). Only combinations that were actually run are shown.</div></div>
</section>
<section id="p2" style="display:none">
 <div class="card"><div class="row">
  <label>Precision <select id="prec"><option value="bf16">bf16</option><option value="fp16">fp16</option><option value="fp8">fp8 weights</option></select></label>
  <label>Offered load <input id="rate" type="range" min="0" max="12" step="1" value="6"><b id="ratev"></b></label></div>
  <div class="grid" id="s2"></div><div class="note" id="n2"></div></div>
 <div class="card"><canvas id="c2" width="1000" height="360"></canvas>
  <div class="note">Open-loop Poisson arrivals, 128 output tokens per request, Qwen2.5-1.5B on vLLM 0.32. Latency is flat until near capacity, then queueing
  makes it explode. The online plateau (about 18 to 22k tokens/s) is set by the Python API stack, not the GPU (engine-only: 28.7k tokens/s).</div></div>
</section>
<section id="p3" style="display:none">
 <div class="card"><div class="row">
  <label>Arrival rate (req/s) <input id="srate" type="range" min="1" max="260" value="96"><b id="srv"></b></label>
  <label>Max sequences in flight <input id="scap" type="range" min="4" max="256" value="256"><b id="scv"></b></label>
  <label>Output tokens <input id="sout" type="range" min="16" max="256" value="128" step="16"><b id="sov"></b></label>
  <button class="act" id="run">Run simulation</button></div>
  <div class="grid" id="s3"></div><div class="note" id="n3"></div></div>
 <div class="card"><canvas id="c3" width="1000" height="360"></canvas>
  <div class="note">Left: one cell per sequence slot (filled = being decoded). Bar: requests waiting. Line: queue length over simulated time.
  Model: each decode step takes <span id="mdl"></span> ms, with N requests in flight (fitted to the measured bf16 load test).
  It tracks measured median latency within about 8% up to 128 req/s, and it knows nothing about the HTTP/Python stack that limits the real server at about 140 req/s.</div></div>
</section>
</main>
<script>
const DATA = __DATA__;
// ==LOGIC==
const SIZES = [8,16,32,64,128,256,512,1024,2048,4096];
const RATES = DATA.serve.bf16.map(r => r.offered);
function lookupTrain(key, batch){
  const cfg = DATA.train[key];
  const hit = cfg.points.find(p => p.batch === batch);
  if (hit) return hit.status === 'ok' ? {state:'ok', mem:hit.mem, sps:hit.sps} : {state:'oom'};
  const ooms = cfg.points.filter(p => p.status === 'oom').map(p => p.batch);
  if (ooms.length && batch > Math.min.apply(null, ooms)) return {state:'oom-inferred', from:Math.min.apply(null, ooms)};
  return {state:'untested'};
}
function fullFineTune(sps){ const s = DATA.full_finetune_samples / sps; return {seconds:s, gpuHours:s/3600}; }
function lookupServe(prec, rate){ return DATA.serve[prec].find(r => r.offered === rate) || null; }
function mulberry32(a){ return function(){ a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function pctile(arr, q){ if (!arr.length) return NaN; const s = arr.slice().sort((x,y)=>x-y); const k=(s.length-1)*q;
  const lo=Math.floor(k), hi=Math.min(lo+1,s.length-1); return s[lo]+(s[hi]-s[lo])*(k-lo); }
function simulate(rate, cap, outLen, duration, seed, frameDt){
  const rng = mulberry32(seed), A = DATA.model.a_ms, C = DATA.model.c_ms, OV = DATA.model.overhead_ms;
  const expo = () => -Math.log(1 - rng()) / rate;
  let t = 0, nxt = expo(), waiting = [], head = 0, active = [], lat = [], frames = [], nextFrame = 0, done = 0, tokens = 0;
  while (t < duration){
    while (nxt <= t){ waiting.push(nxt); nxt += expo(); }
    while (active.length < cap && head < waiting.length){ active.push({rem:outLen, arr:waiting[head++]}); }
    if (!active.length){ t = Math.min(nxt, duration); continue; }
    t += (A + C * active.length) / 1000;
    tokens += active.length;
    for (const r of active) r.rem -= 1;
    const still = [];
    for (const r of active){ if (r.rem <= 0){ lat.push((t - r.arr) * 1000 + OV); done++; } else still.push(r); }
    active = still;
    while (frameDt && nextFrame <= t){ frames.push({t:nextFrame, active:active.length, queue:waiting.length-head, done:done}); nextFrame += frameDt; }
  }
  return {p50:pctile(lat,.5), p99:pctile(lat,.99), done:done, reqPerSec:done/duration, tokPerSec:tokens/duration,
          maxQueue:frames.reduce((m,f)=>Math.max(m,f.queue),0), frames:frames};
}
// ==UI==
const $ = id => document.getElementById(id);
const fmt = (x, d) => (x === null || x === undefined || isNaN(x)) ? 'n/a' : x.toLocaleString('en-US', {maximumFractionDigits:d===undefined?0:d});
function stat(label, value, cls){ return '<div class="stat"><span>'+label+'</span><b class="'+(cls||'')+'">'+value+'</b></div>'; }
function lineChart(cv, series, o){
  const ctx = cv.getContext('2d'), W = cv.width, H = cv.height, L = 62, R = 16, T = 14, B = 40;
  ctx.clearRect(0,0,W,H); ctx.font = '12px sans-serif';
  const xs = [].concat.apply([], series.map(s => s.x)), ys = [].concat.apply([], series.map(s => s.y)).filter(v => v > 0);
  const xmin = o.xmin !== undefined ? o.xmin : Math.min.apply(null, xs), xmax = o.xmax !== undefined ? o.xmax : Math.max.apply(null, xs);
  const ymin = o.ymin !== undefined ? o.ymin : Math.min.apply(null, ys), ymax = o.ymax !== undefined ? o.ymax : Math.max.apply(null, ys);
  const fx = v => o.logX ? Math.log(v) : v, fy = v => o.logY ? Math.log(Math.max(v, 1e-9)) : v;
  const px = v => L + (fx(v) - fx(xmin)) / (fx(xmax) - fx(xmin)) * (W - L - R);
  const py = v => H - B - (fy(v) - fy(ymin)) / (fy(ymax) - fy(ymin)) * (H - T - B);
  ctx.strokeStyle = '#e5e7eb'; ctx.fillStyle = '#6b7280'; ctx.lineWidth = 1;
  (o.xticks || []).forEach(v => { const x = px(v); ctx.beginPath(); ctx.moveTo(x,T); ctx.lineTo(x,H-B); ctx.stroke(); ctx.fillText(String(v), x-8, H-B+16); });
  (o.yticks || []).forEach(v => { const y = py(v); ctx.beginPath(); ctx.moveTo(L,y); ctx.lineTo(W-R,y); ctx.stroke(); ctx.fillText(String(v), 6, y+4); });
  ctx.fillStyle = '#111827'; ctx.fillText(o.xlabel || '', L + (W-L-R)/2 - 60, H-6);
  ctx.save(); ctx.translate(14, T + (H-T-B)/2 + 40); ctx.rotate(-Math.PI/2); ctx.fillText(o.ylabel || '', 0, 0); ctx.restore();
  (o.hlines || []).forEach(h => { const y = py(h.y); ctx.setLineDash([4,4]); ctx.strokeStyle = h.color; ctx.beginPath(); ctx.moveTo(L,y); ctx.lineTo(W-R,y); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = h.color; ctx.fillText(h.label, L+6, y-4); });
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.5; ctx.beginPath();
    s.x.forEach((xv, i) => { const x = px(xv), y = py(s.y[i]); if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.stroke();
    (s.marks || []).forEach(m => { ctx.fillStyle = m.color; ctx.beginPath(); ctx.arc(px(m.x), py(m.y), m.r || 5, 0, 6.2832); ctx.fill(); });
    (s.crosses || []).forEach(m => { ctx.strokeStyle = '#b91c1c'; ctx.lineWidth = 2.5; const x = px(m.x), y = py(m.y);
      ctx.beginPath(); ctx.moveTo(x-6,y-6); ctx.lineTo(x+6,y+6); ctx.moveTo(x+6,y-6); ctx.lineTo(x-6,y+6); ctx.stroke(); });
  });
  if (o.legend){ let ly = T + 14; o.legend.forEach(l => { ctx.fillStyle = l.color; ctx.fillRect(W-R-190, ly-9, 14, 4); ctx.fillStyle = '#111827'; ctx.fillText(l.label, W-R-170, ly-3); ly += 16; }); }
}
// ---- tab 1
function renderTrain(){
  const key = $('cfg').value, batch = SIZES[parseInt($('bs').value, 10)];
  $('bsv').textContent = ' ' + batch;
  const r = lookupTrain(key, batch);
  let html = '', note = '';
  if (r.state === 'ok'){
    const ft = fullFineTune(r.sps);
    html = stat('Peak GPU memory', fmt(r.mem,2)+' GB', r.mem > DATA.cap_gb*0.9 ? 'warn' : 'ok') + stat('Throughput', fmt(r.sps)+' samples/s', 'ok')
      + stat('Full fine-tune (est.)', fmt(ft.seconds)+' s', '') + stat('GPU-hours (est.)', fmt(ft.gpuHours,4), '');
    note = 'Measured on the RTX 4090. The full fine-tune estimate is 31,930 rows x 3 epochs divided by this throughput (training steps only; the real runs add about 5 to 10% for validation).';
  } else if (r.state === 'oom'){
    html = stat('Result', 'OUT OF MEMORY', 'bad'); note = 'Measured: this batch size failed with a CUDA out-of-memory error.';
  } else if (r.state === 'oom-inferred'){
    html = stat('Result', 'would not fit', 'bad'); note = 'Not run: batch ' + r.from + ' already failed for this technique, so larger sizes were not tried.';
  } else {
    html = stat('Result', 'not measured', 'warn'); note = 'This size was not run for this technique (gradient accumulation was swept from 64 up to 2048 and capped there).';
  }
  $('s1').innerHTML = html; $('n1').textContent = note;
  const colors = ['#6b7280','#1f77b4','#2ca02c','#d62728','#9467bd','#ff7f0e','#8c564b'];
  const series = Object.keys(DATA.train).map((k, i) => {
    const okp = DATA.train[k].points.filter(p => p.status === 'ok');
    return {x:okp.map(p=>p.batch), y:okp.map(p=>p.mem), color:colors[i%colors.length], width:k===key?4:1.2,
      crosses:DATA.train[k].points.filter(p=>p.status==='oom').map(p=>({x:p.batch, y:DATA.cap_gb*0.98})),
      marks:(k===key && r.state==='ok') ? [{x:batch, y:r.mem, color:'#111827', r:7}] : []};
  });
  lineChart($('c1'), series, {logX:true, xmin:8, xmax:4096, ymin:0.5, ymax:24, logY:false, xticks:SIZES, yticks:[2,5,10,15,20],
    xlabel:'batch size (effective)', ylabel:'peak GPU memory (GB)', hlines:[{y:DATA.cap_gb, color:'#b91c1c', label:'usable VRAM ' + fmt(DATA.cap_gb,1) + ' GB'}],
    legend:Object.keys(DATA.train).map((k,i)=>({color:colors[i%colors.length], label:DATA.train[k].label.slice(0,26)}))});
}
// ---- tab 2
function renderServe(){
  const prec = $('prec').value, rate = RATES[parseInt($('rate').value, 10)];
  $('ratev').textContent = ' ' + rate + ' req/s';
  const r = lookupServe(prec, rate);
  if (!r){ $('s2').innerHTML = stat('Result', 'no data', 'warn'); $('n2').textContent = 'This precision was not measured at this rate.'; return; }
  const sla = r.goodput >= 0.95 * r.achieved;
  $('s2').innerHTML = stat('End-to-end p50 / p99', fmt(r.e2e_p50)+' / '+fmt(r.e2e_p99)+' ms', r.e2e_p99 > 2000 ? 'bad' : 'ok')
    + stat('Time to first token p50 / p99', fmt(r.ttft_p50)+' / '+fmt(r.ttft_p99)+' ms', r.ttft_p99 > 500 ? 'bad' : 'ok')
    + stat('Time per output token p50', fmt(r.tpot_p50,1)+' ms', '') + stat('Output throughput', fmt(r.tok_s)+' tokens/s', '')
    + stat('Achieved / offered', fmt(r.achieved,1)+' / '+rate+' req/s', r.achieved < 0.9*rate ? 'bad' : 'ok')
    + stat('SLA goodput', fmt(r.goodput,1)+' req/s', sla ? 'ok' : 'bad');
  $('n2').textContent = 'SLA = time to first token <= 500 ms and time per token <= 50 ms. ' + (sla ? 'At least 95% of requests met it.' : 'Fewer than 95% of requests met it: the server is past its useful capacity.');
  const rows = DATA.serve[prec], xs = rows.map(x => x.offered);
  lineChart($('c2'), [
    {x:xs, y:rows.map(x=>x.e2e_p99), color:'#7570b3', width:2.5, marks:[{x:rate, y:r.e2e_p99, color:'#7570b3', r:7}]},
    {x:xs, y:rows.map(x=>x.e2e_p50), color:'#1b9e77', width:2.5, marks:[{x:rate, y:r.e2e_p50, color:'#1b9e77', r:7}]}],
    {logX:true, logY:true, xmin:1, xmax:512, ymin:400, ymax:12000, xticks:[1,4,16,64,256], yticks:[500,1000,2000,5000,10000],
     xlabel:'offered rate (req/s)', ylabel:'end-to-end latency (ms)', legend:[{color:'#7570b3', label:'p99'}, {color:'#1b9e77', label:'p50'}]});
}
// ---- tab 3
let playTimer = null;
function runSim(){
  const rate = parseInt($('srate').value,10), cap = parseInt($('scap').value,10), outLen = parseInt($('sout').value,10);
  $('srv').textContent = ' ' + rate; $('scv').textContent = ' ' + cap; $('sov').textContent = ' ' + outLen;
  $('mdl').textContent = fmt(DATA.model.a_ms,2) + ' + ' + fmt(DATA.model.c_ms,4) + ' x N';
  const res = simulate(rate, cap, outLen, 20, 2122, 0.1);
  let html = stat('Simulated p50 / p99', fmt(res.p50)+' / '+fmt(res.p99)+' ms', res.p99 > 2000 ? 'bad' : 'ok')
    + stat('Served', fmt(res.reqPerSec,1)+' req/s', res.reqPerSec < 0.9*rate ? 'bad' : 'ok') + stat('Output throughput', fmt(res.tokPerSec)+' tokens/s', '')
    + stat('Longest queue', fmt(res.maxQueue)+' requests', res.maxQueue > 20 ? 'bad' : 'ok');
  $('s3').innerHTML = html;
  const meas = outLen === 128 ? DATA.seq_cap_measured[String(cap)] : undefined;
  $('n3').textContent = meas ? ('Measured peak for max-num-seqs ' + cap + ' (real server, short prompts): ' + fmt(meas) + ' tokens/s. The simulation has no HTTP stack, so compare it with the engine-only measurement (' + fmt(DATA.engine_only_tok_s) + ' tokens/s at 256 sequences) at the top end.')
    : 'Change max sequences to 32, 64 or 256 with 128 output tokens to see the measured value for the real server.';
  if (playTimer) clearInterval(playTimer);
  let i = 0; const fr = res.frames;
  const cv = $('c3'), ctx = cv.getContext('2d');
  function draw(){
    const f = fr[Math.min(i, fr.length-1)]; ctx.clearRect(0,0,cv.width,cv.height); ctx.font = '13px sans-serif';
    const cols = Math.min(cap, 32), rowsN = Math.ceil(cap/cols), cw = Math.min(24, 520/cols), ch = Math.min(24, 200/rowsN);
    for (let s = 0; s < cap; s++){ const x = 20 + (s % cols) * cw, y = 40 + Math.floor(s / cols) * ch; ctx.fillStyle = s < f.active ? '#2563eb' : '#e5e7eb'; ctx.fillRect(x, y, cw-2, ch-2); }
    ctx.fillStyle = '#111827'; ctx.fillText('t = ' + f.t.toFixed(1) + ' s   in flight: ' + f.active + ' / ' + cap + '   waiting: ' + f.queue + '   finished: ' + f.done, 20, 22);
    ctx.fillStyle = '#d97706'; ctx.fillRect(20, 270, Math.min(560, f.queue * 4), 16); ctx.fillStyle = '#111827'; ctx.fillText('queue', 20, 306);
    const qmax = Math.max(10, res.maxQueue); ctx.strokeStyle = '#d97706'; ctx.beginPath();
    fr.forEach((g, k) => { const x = 640 + k / fr.length * 340, y = 300 - g.queue / qmax * 220; if (k === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke();
    ctx.fillStyle = '#6b7280'; ctx.fillText('queue length over 20 s (max ' + qmax + ')', 640, 322);
    ctx.strokeStyle = '#111827'; const px = 640 + Math.min(i, fr.length-1) / fr.length * 340; ctx.beginPath(); ctx.moveTo(px, 80); ctx.lineTo(px, 300); ctx.stroke();
  }
  draw();
  playTimer = setInterval(() => { i += 1; draw(); if (i >= fr.length) clearInterval(playTimer); }, 40);
}
// ---- wiring
function show(n){ ['1','2','3'].forEach(k => { $('p'+k).style.display = (k === n) ? 'block' : 'none'; $('t'+k).className = (k === n) ? 'on' : ''; }); }
function init(){
  $('cfg').innerHTML = Object.keys(DATA.train).map(k => '<option value="'+k+'">'+DATA.train[k].label+'</option>').join('');
  ['t1','t2','t3'].forEach(id => $(id).addEventListener('click', () => { show(id.slice(1)); if (id === 't3') runSim(); }));
  $('cfg').addEventListener('change', renderTrain); $('bs').addEventListener('input', renderTrain);
  $('prec').addEventListener('change', renderServe); $('rate').addEventListener('input', renderServe);
  $('run').addEventListener('click', runSim);
  ['srate','scap','sout'].forEach(id => $(id).addEventListener('input', () => { $('srv').textContent=' '+$('srate').value; $('scv').textContent=' '+$('scap').value; $('sov').textContent=' '+$('sout').value; }));
  $('rate').max = String(RATES.length - 1);
  renderTrain(); renderServe();
}
init();
</script></body></html>
"""

os.makedirs("docs", exist_ok=True)
open("docs/simulator.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data)))
print("wrote docs/simulator.html", round(os.path.getsize("docs/simulator.html") / 1024), "KB; model", round(A, 3), round(C, 5))
