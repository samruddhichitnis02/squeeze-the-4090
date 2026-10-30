"""Download the evaluation sets once and save them as local JSONL under data/ (git-ignored; not redistributed).

  MMLU (cais/mmlu, all subjects, test split)       14,042 four-choice questions
  GSM8K (openai/gsm8k, main, test split)            1,319 grade-school math problems
  WikiText-2 (wikitext-2-raw-v1, test split)        raw text for perplexity
"""
import json
import os

from datasets import load_dataset

os.makedirs("data", exist_ok=True)

mm = load_dataset("cais/mmlu", "all", split="test")
with open("data/mmlu_test.jsonl", "w", encoding="utf-8") as f:
    for r in mm:
        f.write(json.dumps(dict(question=r["question"], subject=r["subject"], choices=list(r["choices"]),
                                answer=int(r["answer"]))) + "\n")
print("mmlu", len(mm))

gs = load_dataset("openai/gsm8k", "main", split="test")
with open("data/gsm8k_test.jsonl", "w", encoding="utf-8") as f:
    for r in gs:
        f.write(json.dumps(dict(question=r["question"], answer=r["answer"])) + "\n")
print("gsm8k", len(gs))

try:
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
except Exception:  # noqa: BLE001
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "".join(r["text"] for r in wt)  # standard perplexity practice: concatenate the raw test text
open("data/wikitext2_test.txt", "w", encoding="utf-8").write(text)
print("wikitext2 characters", len(text))
