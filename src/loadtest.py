"""Open-loop load test against a vLLM OpenAI-compatible server.

Requests arrive on a Poisson schedule at a target rate (req/s), independent of how fast the server answers, so
queueing delay shows up in the latency numbers. Streaming is used to measure time-to-first-token (TTFT) and
time-per-output-token (TPOT). Every request generates exactly --max-tokens tokens (EOS suppressed).
"""
import argparse
import asyncio
import json
import os
import random
import time

import aiohttp

sys_path_fix = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, sys_path_fix)
from serve_bench import PROMPTS  # noqa: E402  (reuse the same 32 prompts as the offline benchmark)


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def one_request(session, url, model, prompt, max_tokens, t_sched, results):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
               "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    rec = {"t_sched": t_sched, "ok": False}
    t0 = time.perf_counter()
    rec["t_start"] = t0
    ttft, n_tokens = None, 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}"
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
                ch = obj.get("choices") or []
                if ch and ch[0].get("text") and ttft is None:
                    ttft = time.perf_counter() - t0
        e2e = time.perf_counter() - t0
        rec.update(ok=True, ttft=ttft, e2e=e2e, n_tokens=n_tokens,
                   tpot=(e2e - ttft) / (n_tokens - 1) if (ttft is not None and n_tokens > 1) else None)
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:200]
    results.append(rec)


async def run_rate(args, tokenizer_prompts, rate, n_req):
    url = f"http://{args.host}:{args.port}/v1/completions"
    rng = random.Random(args.seed)
    gaps = [rng.expovariate(rate) for _ in range(n_req)]
    results, tasks = [], []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t_begin = time.perf_counter()
        t_next = 0.0
        for i in range(n_req):
            t_next += gaps[i]
            delay = t_begin + t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            prompt = tokenizer_prompts[i % len(tokenizer_prompts)]
            tasks.append(asyncio.create_task(
                one_request(session, url, args.model, prompt, args.max_tokens, t_next, results)))
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()
    return results, t_end - t_begin


def summarize(results, wall, rate, n_req, args):
    ok = [r for r in results if r["ok"]]
    e2e = [r["e2e"] * 1000 for r in ok]
    ttft = [r["ttft"] * 1000 for r in ok if r["ttft"] is not None]
    tpot = [r["tpot"] * 1000 for r in ok if r["tpot"] is not None]
    tokens = sum(r["n_tokens"] for r in ok)
    # SLA goodput: requests that met both the TTFT and TPOT targets, per second of wall-clock
    good = [r for r in ok if r["ttft"] is not None and r["tpot"] is not None
            and r["ttft"] * 1000 <= args.sla_ttft_ms and r["tpot"] * 1000 <= args.sla_tpot_ms]
    return dict(
        offered_rate=rate, n_requests=n_req, n_ok=len(ok), n_failed=n_req - len(ok), wall_seconds=wall,
        achieved_req_per_sec=len(ok) / wall, output_tokens_per_sec=tokens / wall, total_output_tokens=tokens,
        e2e_ms=dict(p50=pct(e2e, .5), p95=pct(e2e, .95), p99=pct(e2e, .99), mean=sum(e2e) / max(len(e2e), 1)),
        ttft_ms=dict(p50=pct(ttft, .5), p95=pct(ttft, .95), p99=pct(ttft, .99)),
        tpot_ms=dict(p50=pct(tpot, .5), p95=pct(tpot, .95), p99=pct(tpot, .99)),
        sla=dict(ttft_ms=args.sla_ttft_ms, tpot_ms=args.sla_tpot_ms),
        goodput_req_per_sec=len(good) / wall, sla_attainment=len(good) / n_req,
        first_error=next((r["error"] for r in results if not r["ok"]), None))


async def main_async(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True) for p in PROMPTS]
    # warmup so first-request effects (CUDA graphs, caches) are not counted
    await run_rate(args, prompts, 4.0, args.warmup_requests)
    out = []
    for rate in args.rates:
        n_req = int(min(max(args.min_requests, rate * args.seconds_per_rate), args.max_requests))
        results, wall = await run_rate(args, prompts, rate, n_req)
        s = summarize(results, wall, rate, n_req, args)
        out.append(s)
        print("RATE", json.dumps({k: s[k] for k in ["offered_rate", "n_ok", "n_failed", "achieved_req_per_sec",
                                                     "output_tokens_per_sec", "e2e_ms", "ttft_ms", "tpot_ms",
                                                     "goodput_req_per_sec"]}), flush=True)
        # stop pushing once the server is clearly overloaded: latency blow-up or failures
        if s["n_failed"] > 0.1 * n_req or s["e2e_ms"]["p50"] > args.stop_p50_ms:
            print(f"STOP: overload reached at {rate} req/s", flush=True)
            break
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(model=args.model, label=args.label, max_tokens=args.max_tokens, seed=args.seed,
                   rates=out), open(args.out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--label", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
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
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
