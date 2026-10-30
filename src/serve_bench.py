"""Serving benchmark: Hugging Face pipeline vs vLLM on identical work.

One process per configuration (vLLM holds GPU memory, so engines must not share a process).
Every request generates exactly --max-new-tokens tokens (EOS suppressed) so engines do equal work.
Latency unit = one generate call (batch of --batch-size requests). At batch 1 this is per-request latency.
"""
import argparse
import json
import os
import statistics
import time

PROMPTS = [
    "Explain what a KV cache is in a transformer decoder.",
    "Why is autoregressive decoding usually memory-bandwidth bound?",
    "Write a short Python function that reverses a linked list.",
    "Summarize the difference between latency and throughput in model serving.",
    "What is continuous batching and why does it help LLM serving?",
    "Give three tips for reducing GPU memory use during fine-tuning.",
    "Explain gradient checkpointing in simple terms.",
    "What does mixed precision training change about numerical range?",
    "Describe how paged attention manages memory.",
    "Why do tail latencies (p99) matter more than averages for user-facing apps?",
    "Explain int4 weight-only quantization and its main tradeoff.",
    "What is the difference between prefill and decode phases?",
    "Write a haiku about a GPU running out of memory.",
    "Compare FP16, BF16 and FP8 in two sentences each.",
    "How does speculative decoding speed up generation?",
    "Explain what tokens per second measures and what it hides.",
    "List four metrics you would track for an inference service.",
    "What is a roofline model in performance engineering?",
    "Describe batch size's effect on GPU utilization.",
    "Explain why the first request to a model is slower than later ones.",
    "Write a SQL query that finds the top 5 customers by total spend.",
    "What is prefix caching and when does it help?",
    "Explain tensor cores to someone who knows basic CPU architecture.",
    "Why can larger batches hurt per-request latency?",
    "Explain the idea behind FlashAttention.",
    "What is the purpose of a warmup run when benchmarking?",
    "Describe the memory cost of optimizer states in Adam.",
    "Explain what an out-of-memory error in CUDA means.",
    "Give a short example of prompt engineering for classification.",
    "How do you compute GPU-hours for a training job?",
    "Explain the tradeoff between model quality and quantization level.",
    "What are common causes of GPU underutilization during inference?",
]


def build_prompts(tokenizer, n):
    texts = []
    for i in range(n):
        msg = [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]
        texts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    return texts


def pct(values, q):
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm"], required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--num-batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--out", default="results/step2/raw")
    ap.add_argument("--quantization", default=None, help="vLLM online quantization, e.g. fp8")
    ap.add_argument("--tag", default="", help="suffix for the result filename")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    B = a.batch_size
    all_prompts = build_prompts(tok, B * (a.num_batches + a.warmup))
    batches = [all_prompts[i * B:(i + 1) * B] for i in range(a.num_batches + a.warmup)]

    if a.engine == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
                  max_model_len=1024, seed=2122, max_num_seqs=256, quantization=a.quantization)
        sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, ignore_eos=True)

        def run(batch):
            outs = llm.generate(batch, sp, use_tqdm=False)
            return sum(len(o.outputs[0].token_ids) for o in outs)
    else:
        import torch
        from transformers import pipeline
        try:
            pipe = pipeline("text-generation", model=a.model, dtype=torch.bfloat16, device=0)
        except TypeError:
            pipe = pipeline("text-generation", model=a.model, torch_dtype=torch.bfloat16, device=0)
        pipe.tokenizer.padding_side = "left"
        if pipe.tokenizer.pad_token is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        torch.cuda.reset_peak_memory_stats()

        def run(batch):
            outs = pipe(batch, max_new_tokens=a.max_new_tokens, min_new_tokens=a.max_new_tokens,
                        do_sample=False, return_full_text=False, batch_size=len(batch))
            # Token count is assumed, not measured: min_new_tokens == max_new_tokens forces exactly this
            # many tokens per request. (The vLLM path counts real output tokens.)
            return len(batch) * a.max_new_tokens

    for b in batches[:a.warmup]:
        run(b)

    lat_ms, total_tokens = [], 0
    t0 = time.perf_counter()
    for b in batches[a.warmup:]:
        t = time.perf_counter()
        total_tokens += run(b)
        lat_ms.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - t0

    n_req = B * a.num_batches
    res = dict(
        engine=a.engine, batch_size=B, model=a.model, max_new_tokens=a.max_new_tokens,
        num_batches=a.num_batches, num_requests=n_req, total_generated_tokens=total_tokens,
        wall_seconds=wall, tokens_per_sec=total_tokens / wall, requests_per_sec=n_req / wall,
        latency_ms=dict(p50=pct(lat_ms, 0.50), p95=pct(lat_ms, 0.95), mean=statistics.mean(lat_ms),
                        min=min(lat_ms), max=max(lat_ms)),
        per_call_latency_ms=lat_ms)
    if a.engine == "hf":
        import torch
        res["peak_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.engine}_bs{B}{a.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "per_call_latency_ms"}))


if __name__ == "__main__":
    main()
