"""vLLM inference demo: run your own prompts through a chosen Hugging Face model."""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="prompts/my_prompts.txt")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2122)
    ap.add_argument("--out", default="results/step2/demo")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=2048, seed=a.seed)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed)

    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    records = []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        records.append(dict(prompt=p, response=c.text.strip(), prompt_tokens=len(o.prompt_token_ids),
                            generated_tokens=len(c.token_ids), finish_reason=c.finish_reason))
    total = sum(r["generated_tokens"] for r in records)
    res = dict(model=a.model, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
               seed=a.seed, seconds=dt, total_generated_tokens=total, tokens_per_sec=total / dt, records=records)

    os.makedirs(a.out, exist_ok=True)
    slug = a.model.replace("/", "__")
    json.dump(res, open(f"{a.out}/{slug}.json", "w"), indent=2)
    md = [f"# vLLM demo: {a.model}", "",
          f"temperature={a.temperature}, top_p={a.top_p}, max_tokens={a.max_tokens}, seed={a.seed}",
          f"{len(records)} prompts in one batch: {total} tokens in {dt:.2f}s ({total / dt:.0f} tokens/sec)", ""]
    for r in records:
        md += ["## Prompt", r["prompt"], "", f"## Response ({r['generated_tokens']} tokens, {r['finish_reason']})",
               r["response"], "", "---", ""]
    open(f"{a.out}/{slug}.md", "w", encoding="utf-8").write("\n".join(md))
    print(f"DEMO_DONE {a.model} tokens={total} sec={dt:.2f}")


if __name__ == "__main__":
    main()
