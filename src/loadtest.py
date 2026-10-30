"""Open-loop load test against a vLLM OpenAI-compatible server.

Requests arrive on a Poisson schedule at a target rate (req/s), independent of how fast the server answers, so
queueing delay shows up in the latency numbers. Streaming is used to measure time-to-first-token (TTFT) and
time-per-output-token (TPOT). Every request generates exactly --max-tokens tokens (EOS suppressed).

--procs N splits the load over N client processes (each with its own event loop and 1/N of the rate) so a single
Python process is not the bottleneck. --workload chooses short chat prompts or a long shared-prefix prompt.
"""
import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_bench import PROMPTS  # noqa: E402  (the same 32 questions as the offline benchmark)

DOC_TOKENS = 900
DOC_SENTENCES = [
    "A decoder-only transformer generates one token per forward pass and keeps the keys and values of earlier tokens in a KV cache.",
    "During prefill the whole prompt is processed in parallel, so the matrix multiplications are large and the GPU is limited by compute.",
    "During decode each step multiplies a small batch of vectors by every weight matrix, so the GPU is limited by memory bandwidth.",
    "Continuous batching lets a server add new requests to the running batch at every step instead of waiting for the batch to finish.",
    "Paged attention stores the KV cache in fixed-size blocks so memory is not wasted on sequences of unknown final length.",
    "Prefix caching reuses the KV blocks of a prompt prefix that was seen before, so a shared document is prefilled only once.",
    "Chunked prefill splits a long prompt into pieces and mixes them with decode steps so running requests are not stalled for long.",
    "Quantizing weights to eight bits halves the bytes read per decode step, which helps most when the batch is small.",
    "Storing the KV cache in eight bits halves its memory, which allows more concurrent sequences or longer contexts.",
    "Tail latency matters more than the average for interactive applications because a few slow responses dominate the user experience.",
    "Goodput counts only the requests that meet a latency target, which is a better measure of useful capacity than raw throughput.",
    "Queueing delay grows sharply as the arrival rate approaches the service rate, which is why latency curves bend upward near capacity.",
]


def build_prompts(tok, workload):
    """Return the list of prompt strings (chat template applied) for a workload."""
    if workload == "short":
        return [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True) for q in PROMPTS]
    # long shared prefix: ~DOC_TOKENS tokens of reference notes identical for every request, then a short unique question
    text = " ".join(f"({i + 1}) {DOC_SENTENCES[i % len(DOC_SENTENCES)]}" for i in range(200))
    ids = tok.encode(text)[:DOC_TOKENS]
    doc = tok.decode(ids)
    return [tok.apply_chat_template(
        [{"role": "user", "content": f"Use the reference notes below to answer.\n\n{doc}\n\nQuestion: {q}"}],
        tokenize=False, add_generation_prompt=True) for q in PROMPTS]


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


NO_STREAM = False  # set from --no-stream; capacity probe only (no TTFT/TPOT)


async def one_request(session, url, model, prompt, max_tokens, t_sched, results):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
               "stream": not NO_STREAM, "ignore_eos": True}
    if not NO_STREAM:
        payload["stream_options"] = {"include_usage": True}
    rec = {"t_sched": t_sched, "ok": False}
    t0 = time.perf_counter()
    ttft, n_tokens, n_prompt = None, 0, 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}"
                results.append(rec)
                return
            if NO_STREAM:
                obj = await resp.json()
                e2e = time.perf_counter() - t0
                rec.update(ok=True, ttft=None, e2e=e2e, n_tokens=obj["usage"]["completion_tokens"],
                           n_prompt=obj["usage"].get("prompt_tokens", 0), tpot=None)
                results.append(rec)
                return
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    n_tokens = obj["usage"]["completion_tokens"]
                    n_prompt = obj["usage"].get("prompt_tokens", 0)
                ch = obj.get("choices") or []
                if ch and ch[0].get("text") and ttft is None:
                    ttft = time.perf_counter() - t0
        e2e = time.perf_counter() - t0
        rec.update(ok=True, ttft=ttft, e2e=e2e, n_tokens=n_tokens, n_prompt=n_prompt,
                   tpot=(e2e - ttft) / (n_tokens - 1) if (ttft is not None and n_tokens > 1) else None)
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:200]
    results.append(rec)


async def run_rate(host, port, model, max_tokens, prompts, rate, n_req, seed):
    url = f"http://{host}:{port}/v1/completions"
    rng = random.Random(seed)
    gaps = [rng.expovariate(rate) for _ in range(n_req)]
    results, tasks = [], []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t_begin = time.perf_counter()
        t_next = 0.0
        for i in range(n_req):
            t_next += gaps[i]
            delay = t_begin + t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(
                one_request(session, url, model, prompts[(i + seed) % len(prompts)], max_tokens, t_next, results)))
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()
    return results, t_end - t_begin


def _worker(cfg):
    global NO_STREAM
    NO_STREAM = cfg.get("no_stream", False)
    return asyncio.run(run_rate(cfg["host"], cfg["port"], cfg["model"], cfg["max_tokens"], cfg["prompts"],
                                cfg["rate"], cfg["n_req"], cfg["seed"]))


def run_rate_mp(pool, args, prompts, rate, n_req, seed_offset=0):
    """Split one target rate over args.procs client processes and merge their results."""
    p = args.procs
    n_per = math.ceil(n_req / p)
    cfgs = [dict(host=args.host, port=args.port, model=args.model, max_tokens=args.max_tokens, prompts=prompts,
                 rate=rate / p, n_req=n_per, seed=args.seed + 1000 * seed_offset + i,
                 no_stream=args.no_stream) for i in range(p)]
    parts = list(pool.map(_worker, cfgs))
    results = [r for res, _ in parts for r in res]
    return results, max(w for _, w in parts), n_per * p


def summarize(results, wall, rate, n_req, args):
    ok = [r for r in results if r["ok"]]
    e2e = [r["e2e"] * 1000 for r in ok]
    ttft = [r["ttft"] * 1000 for r in ok if r["ttft"] is not None]
    tpot = [r["tpot"] * 1000 for r in ok if r["tpot"] is not None]
    tokens = sum(r["n_tokens"] for r in ok)
    good = [r for r in ok if r["ttft"] is not None and r["tpot"] is not None
            and r["ttft"] * 1000 <= args.sla_ttft_ms and r["tpot"] * 1000 <= args.sla_tpot_ms]
    return dict(
        offered_rate=rate, n_requests=n_req, n_ok=len(ok), n_failed=n_req - len(ok), wall_seconds=wall,
        achieved_req_per_sec=len(ok) / wall, output_tokens_per_sec=tokens / wall, total_output_tokens=tokens,
        mean_prompt_tokens=sum(r["n_prompt"] for r in ok) / max(len(ok), 1),
        e2e_ms=dict(p50=pct(e2e, .5), p95=pct(e2e, .95), p99=pct(e2e, .99), mean=sum(e2e) / max(len(e2e), 1)),
        ttft_ms=dict(p50=pct(ttft, .5), p95=pct(ttft, .95), p99=pct(ttft, .99)),
        tpot_ms=dict(p50=pct(tpot, .5), p95=pct(tpot, .95), p99=pct(tpot, .99)),
        sla=dict(ttft_ms=args.sla_ttft_ms, tpot_ms=args.sla_tpot_ms),
        goodput_req_per_sec=len(good) / wall, sla_attainment=len(good) / n_req,
        first_error=next((r["error"] for r in results if not r["ok"]), None))


def main_sync(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tok, args.workload)
    print(f"workload={args.workload} prompts={len(prompts)} procs={args.procs}", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=args.procs, mp_context=get_context("spawn")) as pool:
        run_rate_mp(pool, args, prompts, 4.0, max(args.warmup_requests, args.procs), seed_offset=99)  # warmup
        for k, rate in enumerate(args.rates):
            n_req = int(min(max(args.min_requests, rate * args.seconds_per_rate), args.max_requests))
            results, wall, n_total = run_rate_mp(pool, args, prompts, rate, n_req, seed_offset=k)
            s = summarize(results, wall, rate, n_total, args)
            out.append(s)
            print("RATE", json.dumps({k_: s[k_] for k_ in ["offered_rate", "n_ok", "n_failed", "achieved_req_per_sec",
                                                            "output_tokens_per_sec", "mean_prompt_tokens", "e2e_ms",
                                                            "ttft_ms", "tpot_ms", "goodput_req_per_sec"]}), flush=True)
            if s["n_failed"] > 0.1 * n_total or s["e2e_ms"]["p50"] > args.stop_p50_ms:
                print(f"STOP: overload reached at {rate} req/s", flush=True)
                break
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(model=args.model, label=args.label, workload=args.workload, procs=args.procs,
                   max_tokens=args.max_tokens, seed=args.seed, rates=out), open(args.out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--label", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workload", choices=["short", "longprefix"], default="short")
    ap.add_argument("--procs", type=int, default=1, help="client processes sharing the offered rate")
    ap.add_argument("--no-stream", action="store_true", help="non-streaming requests: capacity probe, no TTFT/TPOT")
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--seconds-per-rate", type=float, default=20)
    ap.add_argument("--min-requests", type=int, default=60)
    ap.add_argument("--max-requests", type=int, default=1500)
    ap.add_argument("--warmup-requests", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2122)
    ap.add_argument("--sla-ttft-ms", type=float, default=500.0)
    ap.add_argument("--sla-tpot-ms", type=float, default=50.0)
    ap.add_argument("--stop-p50-ms", type=float, default=20000.0)
    ap.add_argument("--out", required=True)
    main_sync(ap.parse_args())


if __name__ == "__main__":
    main()
