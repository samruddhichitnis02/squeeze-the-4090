"""HW4.a: BERT sentiment fine-tune on tweet_eval (sentiment) with full measurement.

Settings are the "HW4 settings" that HW4.5 later uses as the baseline:
fp32, AdamW lr=2e-5, batch 32, max_len 64, 4 epochs.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

p = argparse.ArgumentParser()
p.add_argument("--slice", type=int, default=0, dest="slice_start")
p.add_argument("--n_rows", type=int, default=5000)
p.add_argument("--seed", type=int, default=2122)
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--max_len", type=int, default=64)
p.add_argument("--model", default="bert-base-uncased")
p.add_argument("--out", default="outputs")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
LABELS = ["negative", "neutral", "positive"]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed(args.seed)
dev = torch.device("cuda")

# ---------- 1. data: up to 5000 consecutive rows starting at SLICE ----------
full = load_dataset("cardiffnlp/tweet_eval", "sentiment")["train"]
start = args.slice_start
end = min(start + args.n_rows, len(full))  # exclusive
rows = full.select(range(start, end))
print(f"ROW RANGE: train-split rows [{start}, {end - 1}] inclusive, n={len(rows)}")
texts, labels = list(rows["text"]), list(rows["label"])

idx = np.arange(len(texts))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=args.seed, stratify=None)
va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=args.seed)
print(f"split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

tok = BertTokenizer.from_pretrained(args.model)


def make_ds(ix):
    enc = tok([texts[i] for i in ix], padding="max_length", truncation=True,
              max_length=args.max_len, return_tensors="pt")
    y = torch.tensor([labels[i] for i in ix])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


train_ds, val_ds, test_ds = make_ds(tr_idx), make_ds(va_idx), make_ds(te_idx)
g = torch.Generator().manual_seed(args.seed)
train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
val_dl = DataLoader(val_ds, batch_size=128)
test_dl = DataLoader(test_ds, batch_size=128)

# ---------- 2. model ----------
model = BertForSequenceClassification.from_pretrained(args.model, num_labels=3).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)


@torch.no_grad()
def evaluate(dl):
    model.eval()
    tot, n, correct = 0.0, 0, 0
    preds, gold = [], []
    for ids, am, y in dl:
        ids, am, y = ids.to(dev), am.to(dev), y.to(dev)
        out = model(input_ids=ids, attention_mask=am, labels=y)
        tot += out.loss.item() * y.size(0)
        n += y.size(0)
        pr = out.logits.argmax(-1)
        correct += (pr == y).sum().item()
        preds += pr.cpu().tolist()
        gold += y.cpu().tolist()
    return tot / n, correct / n, preds, gold


# ---------- 3. train ----------
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
t_start = time.perf_counter()
history, best = [], {"val_acc": -1, "epoch": 0, "state": None}
for ep in range(1, args.epochs + 1):
    model.train()
    t_ep = time.perf_counter()
    run, n = 0.0, 0
    for ids, am, y in train_dl:
        ids, am, y = ids.to(dev), am.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=am, labels=y).loss
        loss.backward()
        opt.step()
        run += loss.item() * y.size(0)
        n += y.size(0)
    torch.cuda.synchronize()
    ep_time = time.perf_counter() - t_ep
    vl, va, _, _ = evaluate(val_dl)
    rec = dict(epoch=ep, train_loss=run / n, val_loss=vl, val_acc=va, epoch_seconds=ep_time)
    history.append(rec)
    print(json.dumps(rec))
    if va > best["val_acc"]:
        best = {"val_acc": va, "epoch": ep,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
torch.cuda.synchronize()
train_seconds = time.perf_counter() - t_start
peak_gb = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

best_val_loss_epoch = min(history, key=lambda r: r["val_loss"])["epoch"]
extra_epochs = args.epochs - best["epoch"]
extra_seconds = sum(r["epoch_seconds"] for r in history if r["epoch"] > best["epoch"])

# ---------- 4. test with best-val-accuracy checkpoint ----------
model.load_state_dict(best["state"])
tl, ta, preds, gold = evaluate(test_dl)
cm = confusion_matrix(gold, preds, labels=[0, 1, 2])
prec, rec_, f1, sup = precision_recall_fscore_support(gold, preds, labels=[0, 1, 2], zero_division=0)
worst = int(np.argmin(f1))
te_texts = [texts[i] for i in te_idx]
mis = [(te_texts[i], LABELS[gold[i]], LABELS[preds[i]])
       for i in range(len(gold)) if gold[i] == worst and preds[i] != worst]
rng = random.Random(args.seed)
mis_sample = rng.sample(mis, min(5, len(mis)))

# ---------- 5. save + reload for custom inference ----------
save_dir = os.path.join(args.out, "bert_sentiment")
model.save_pretrained(save_dir)
tok.save_pretrained(save_dir)
del model
torch.cuda.empty_cache()
reloaded = BertForSequenceClassification.from_pretrained(save_dir).to(dev).eval()
tok2 = BertTokenizer.from_pretrained(save_dir)


def predict(sentence):
    enc = tok2(sentence, return_tensors="pt", truncation=True, max_length=args.max_len).to(dev)
    with torch.no_grad():
        return LABELS[reloaded(**enc).logits.argmax(-1).item()]


demo = {s: predict(s) for s in [
    "I absolutely love this new phone, best purchase ever!",
    "Worst customer service I have ever had. Never again.",
    "The meeting is scheduled for 3pm tomorrow."]}

results = dict(
    row_range=[start, end - 1], n_rows=len(rows), seed=args.seed,
    split=dict(train=len(tr_idx), val=len(va_idx), test=len(te_idx)),
    settings=dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  max_len=args.max_len, precision="fp32", model=args.model),
    history=history,
    best_val_acc_epoch=best["epoch"], best_val_loss_epoch=best_val_loss_epoch,
    epochs_past_best=extra_epochs, seconds_past_best=extra_seconds,
    test_loss=tl, test_acc=ta,
    confusion_matrix=cm.tolist(),
    per_class={LABELS[i]: dict(precision=float(prec[i]), recall=float(rec_[i]),
                               f1=float(f1[i]), support=int(sup[i])) for i in range(3)},
    worst_class=LABELS[worst], misclassified_examples=mis_sample,
    n_misclassified_in_worst=len(mis),
    peak_mem_allocated_gb=peak_gb, peak_mem_reserved_gb=peak_reserved_gb,
    train_wallclock_seconds=train_seconds,
    gpu=torch.cuda.get_device_name(0), custom_inference_demo=demo)
with open(os.path.join(args.out, "hw4a_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("RESULTS_JSON_WRITTEN")
print(json.dumps({k: results[k] for k in ["best_val_acc_epoch", "best_val_loss_epoch",
      "epochs_past_best", "seconds_past_best", "test_acc", "peak_mem_allocated_gb",
      "train_wallclock_seconds", "worst_class", "custom_inference_demo"]}, indent=2))
