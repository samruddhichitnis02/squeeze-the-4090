"""Quality + speed evaluation of one vLLM configuration (one process = one engine = one config).

Measures, in this order:
  speed  : batch-1 call latency (128 tokens) and engine-only throughput at 256 concurrent sequences
  ppl    : WikiText-2 perplexity (teacher-forced, non-overlapping windows) via prompt_logprobs
  mmlu   : zero-shot accuracy; the answer is the arg-max over the four letter tokens at the first generated position
  gsm8k  : zero-shot chain-of-thought accuracy, greedy, final number extracted from the text
Per-question outcomes are saved so configurations can be compared question by question (paired).
"""
import argparse
import gzip
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_mmlu_prompt(r):
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(r["choices"]))
    subj = r["subject"].replace("_", " ")
    return (f"The following is a multiple choice question about {subj}. Answer with the letter of the correct option only.\n\n"
            f"{r['question']}\n{body}\nAnswer:")


GSM_INSTR = ("Solve the following math problem step by step. At the end, write the final answer on its own line as "
             "'The answer is N' where N is a single number.\n\n")


def extract_number(text):
    m = re.findall(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I)
    if not m:
        m = re.findall(r"\\boxed\{\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not m:
        return None
    s = m[-1].replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gold_number(ans):
    return float(ans.split("####")[-1].strip().replace(",", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--limit-mmlu", type=int, default=0, help="0 = all 14,042")
    ap.add_argument("--limit-gsm", type=int, default=0, help="0 = all 1,319")
    ap.add_argument("--ppl-tokens", type=int, default=0, help="0 = whole WikiText-2 test set")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--out", default="results/step8/raw")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from serve_bench import PROMPTS

    kw = {}
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    llm = LLM(model=a.model, dtype=a.dtype, quantization=a.quantization, kv_cache_dtype=a.kv_cache_dtype,
              gpu_memory_utilization=0.6, max_model_len=2048, max_num_seqs=256, seed=2122, **kw)
    tok = llm.get_tokenizer()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    res = dict(label=a.label, model=a.model, dtype=a.dtype, quantization=a.quantization, kv_cache_dtype=a.kv_cache_dtype,
               attention_backend=a.attention_backend or "default")
    os.makedirs(a.out, exist_ok=True)

    # ---------------- speed
    sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    prompts = [chat(PROMPTS[i % len(PROMPTS)]) for i in range(256)]
    for _ in range(3):
        llm.generate(prompts[:1], sp, use_tqdm=False)
    lat = []
    for i in range(12):
        t = time.perf_counter()
        llm.generate([prompts[i]], sp, use_tqdm=False)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    llm.generate(prompts, sp, use_tqdm=False)  # warmup at 256
    tps = []
    for _ in range(4):
        t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        tps.append(sum(len(o.outputs[0].token_ids) for o in outs) / dt)
    res["speed"] = dict(batch1_latency_p50_ms=lat[len(lat) // 2], batch1_ms_per_token=lat[len(lat) // 2] / 128,
                        batch256_tokens_per_sec=sorted(tps)[len(tps) // 2])
    print("SPEED", json.dumps(res["speed"]), flush=True)

    # ---------------- perplexity
    text = open("data/wikitext2_test.txt", encoding="utf-8").read()
    ids = tok.encode(text)
    if a.ppl_tokens:
        ids = ids[:a.ppl_tokens]
    windows = [ids[i:i + a.window] for i in range(0, len(ids) - 1, a.window) if len(ids[i:i + a.window]) > 1]
    spp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    nll, count = 0.0, 0
    for s in range(0, len(windows), 64):
        batch = windows[s:s + 64]
        outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in batch], spp, use_tqdm=False)
        for w, o in zip(batch, outs):
            for pos in range(1, len(w)):
                lp = o.prompt_logprobs[pos][w[pos]].logprob
                nll -= lp
                count += 1
    res["ppl"] = dict(perplexity=math.exp(nll / count), tokens_scored=count, windows=len(windows), window=a.window)
    print("PPL", json.dumps(res["ppl"]), flush=True)

    # ---------------- MMLU
    rows = [json.loads(l) for l in open("data/mmlu_test.jsonl", encoding="utf-8")]
    if a.limit_mmlu:
        step = max(1, len(rows) // a.limit_mmlu)
        rows = rows[::step][:a.limit_mmlu]
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
    spm = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=letter_ids)
    outs = llm.generate([chat(build_mmlu_prompt(r)) for r in rows], spm, use_tqdm=False)
    correct = []
    for r, o in zip(rows, outs):
        pred = letter_ids.index(o.outputs[0].token_ids[0]) if o.outputs[0].token_ids[0] in letter_ids else -1
        correct.append(int(pred == r["answer"]))
    res["mmlu"] = dict(n=len(rows), accuracy=sum(correct) / len(correct), correct="".join(map(str, correct)))
    print("MMLU", len(rows), round(res["mmlu"]["accuracy"], 4), flush=True)

    # ---------------- GSM8K
    rows = [json.loads(l) for l in open("data/gsm8k_test.jsonl", encoding="utf-8")]
    if a.limit_gsm:
        rows = rows[:a.limit_gsm]
    spg = SamplingParams(temperature=0.0, max_tokens=512)
    t = time.perf_counter()
    outs = llm.generate([chat(GSM_INSTR + r["question"]) for r in rows], spg, use_tqdm=False)
    gdt = time.perf_counter() - t
    correct, texts, gen_tokens = [], [], 0
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text
        pred = extract_number(txt)
        correct.append(int(pred is not None and abs(pred - gold_number(r["answer"])) < 1e-6))
        texts.append(txt)
        gen_tokens += len(o.outputs[0].token_ids)
    res["gsm8k"] = dict(n=len(rows), accuracy=sum(correct) / len(correct), correct="".join(map(str, correct)),
                        mean_generated_tokens=gen_tokens / len(rows), seconds=gdt)
    print("GSM8K", len(rows), round(res["gsm8k"]["accuracy"], 4), flush=True)

    json.dump(res, open(os.path.join(a.out, f"{a.label}.json"), "w"), indent=1)
    with gzip.open(os.path.join(a.out, f"{a.label}.gsm8k_outputs.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(texts, f)
    print("EVAL_DONE", a.label, flush=True)


if __name__ == "__main__":
    main()
