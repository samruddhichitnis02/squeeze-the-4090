"""Batch-size scaling for the BERT fine-tune: one batch size per process (clean memory state).

Uses the same data subset, split, seed, model and optimizer as train_bert.py (fp32 AdamW, lr 2e-5, max_len 64).
Batches larger than the 3,500-row train set are built by tiling/resampling the train rows. This measures
step throughput and memory, not convergence. Exit code 3 means out-of-memory (result still written).
"""
import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer

import techniques

p = argparse.ArgumentParser()
p.add_argument("--batch-size", type=int, required=True)
p.add_argument("--warmup-steps", type=int, default=3)
p.add_argument("--timed-steps", type=int, default=20)
p.add_argument("--seed", type=int, default=2122)
p.add_argument("--slice", type=int, default=0, dest="slice_start")
p.add_argument("--n-rows", type=int, default=5000)
p.add_argument("--max-len", type=int, default=64)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--model", default="bert-base-uncased")
p.add_argument("--out", default="results/step3/raw")
p.add_argument("--micro-batch", type=int, default=None,
               help="gradient accumulation: split each batch into micro-batches of this size")
techniques.add_args(p)
a = p.parse_args()

random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)
dev = torch.device("cuda")
B = a.batch_size
micro = a.micro_batch or B
assert B % micro == 0, "batch size must be divisible by micro-batch"

# Windows/WDDM can silently spill GPU allocations into system RAM instead of raising OOM, which turns the
# memory wall into an extremely slow run. Cap PyTorch's allocator at 95% of the VRAM actually free at start
# (Windows already holds some for the display) so exceeding it raises a real OutOfMemoryError.
torch.zeros(1, device=dev)  # initialise the CUDA context first
free_b, total_b = torch.cuda.mem_get_info()
usable_frac = 0.95 * free_b / total_b
torch.cuda.set_per_process_memory_fraction(usable_frac)
usable_gb = usable_frac * total_b / 1024**3

# same data and split as train_bert.py
full = load_dataset("cardiffnlp/tweet_eval", "sentiment")["train"]
end = min(a.slice_start + a.n_rows, len(full))
rows = full.select(range(a.slice_start, end))
texts, labels = list(rows["text"]), list(rows["label"])
idx = np.arange(len(texts))
tr_idx, _ = train_test_split(idx, test_size=0.30, random_state=a.seed)
tok = BertTokenizer.from_pretrained(a.model)
enc = tok([texts[i] for i in tr_idx], padding="max_length", truncation=True,
          max_length=a.max_len, return_tensors="pt")
ids_all, am_all = enc["input_ids"], enc["attention_mask"]
y_all = torch.tensor([labels[i] for i in tr_idx])
n_train = len(tr_idx)

g = torch.Generator().manual_seed(a.seed)


def next_batch():
    # sample with replacement so any batch size can be produced from the 3,500 train rows
    ix = torch.randint(0, n_train, (B,), generator=g)
    return ids_all[ix], am_all[ix], y_all[ix]


torch.cuda.reset_peak_memory_stats()
result = dict(batch_size=B, micro_batch=micro, accum_steps=B // micro, technique=techniques.describe(a),
              model=a.model, precision="bf16-autocast" if a.bf16 else "fp32", max_len=a.max_len, lr=a.lr,
              train_rows=n_train, warmup_steps=a.warmup_steps, timed_steps=a.timed_steps,
              gpu=torch.cuda.get_device_name(0),
              gpu_total_gb=torch.cuda.get_device_properties(0).total_memory / 1024**3,
              free_gb_at_start=free_b / 1024**3, usable_gb=usable_gb)
status = "ok"
try:
    model = BertForSequenceClassification.from_pretrained(a.model, num_labels=3).to(dev)
    techniques.prepare_model(model, a)
    opt = techniques.make_optimizer(model, a, a.lr)
    model.train()

    def step():
        ids_b, am_b, y_b = next_batch()
        opt.zero_grad(set_to_none=True)
        for j in range(0, B, micro):
            ids = ids_b[j:j + micro].to(dev)
            am = am_b[j:j + micro].to(dev)
            y = y_b[j:j + micro].to(dev)
            with techniques.autocast_ctx(a):
                loss = model(input_ids=ids, attention_mask=am, labels=y).loss
            (loss * (micro / B)).backward()
        opt.step()
        return loss

    for _ in range(a.warmup_steps):
        step()
    torch.cuda.synchronize()
    step_times, last_loss = [], None
    for _ in range(a.timed_steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        last_loss = step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t)
    total = sum(step_times)
    result.update(
        samples_per_sec=B * a.timed_steps / total,
        mean_step_ms=1000 * total / a.timed_steps,
        median_step_ms=1000 * sorted(step_times)[len(step_times) // 2],
        est_epoch_seconds=n_train / (B * a.timed_steps / total),
        peak_mem_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
        peak_mem_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
        final_loss=float(last_loss.item()))
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    if "out of memory" not in str(e).lower():
        raise
    status = "oom"
    result.update(error=str(e).splitlines()[0][:300],
                  peak_mem_allocated_gb_at_failure=torch.cuda.max_memory_allocated() / 1024**3)
    gc.collect()

result["status"] = status
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, f"bs{B:06d}.json"), "w") as f:
    json.dump(result, f, indent=2)
print("RESULT", json.dumps(result))
sys.exit(3 if status == "oom" else 0)
