"""Turn hw4a_results.json into a readable SUMMARY.md next to it."""
import json
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step1"
r = json.load(open(f"{out_dir}/hw4a_results.json"))

L = []
L.append("# HW4.a Step 1: BERT fine-tune summary\n")
L.append(f"- GPU: {r['gpu']}")
L.append(f"- Row range used (train split, inclusive): {r['row_range'][0]} to {r['row_range'][1]} (n={r['n_rows']})")
L.append(f"- Seed: {r['seed']}")
L.append(f"- Split: train={r['split']['train']} val={r['split']['val']} test={r['split']['test']}")
L.append(f"- Settings: {json.dumps(r['settings'])}\n")

L.append("## Per-epoch\n")
L.append("| Epoch | Train loss | Val loss | Val acc | Epoch seconds |")
L.append("|---|---|---|---|---|")
for h in r["history"]:
    L.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | "
             f"{h['val_acc']:.4f} | {h['epoch_seconds']:.2f} |")
L.append("")
L.append(f"- Best val accuracy at epoch {r['best_val_acc_epoch']}; lowest val loss at epoch {r['best_val_loss_epoch']}")
L.append(f"- Epochs trained past best: {r['epochs_past_best']} (cost {r['seconds_past_best']:.1f} s)")
L.append(f"- Test loss {r['test_loss']:.4f}, test accuracy {r['test_acc']:.4f} (best-val-accuracy checkpoint)\n")

L.append("## Confusion matrix (rows true, cols predicted: negative, neutral, positive)\n")
L.append("```")
for row in r["confusion_matrix"]:
    L.append(str(row))
L.append("```\n")

L.append("## Per-class metrics\n")
L.append("| Class | Precision | Recall | F1 | Support |")
L.append("|---|---|---|---|---|")
for k, v in r["per_class"].items():
    L.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
L.append("")
L.append(f"Worst class (lowest F1): **{r['worst_class']}** "
         f"({r['n_misclassified_in_worst']} misclassified in test)\n")
L.append("## Five misclassified examples from the worst class\n")
for text, gold, pred in r["misclassified_examples"]:
    L.append(f"- true={gold}, predicted={pred}: {text}")
L.append("")

L.append("## HW4.5 baseline measurements\n")
L.append(f"- Peak GPU memory allocated: {r['peak_mem_allocated_gb']:.3f} GB")
L.append(f"- Peak GPU memory reserved: {r['peak_mem_reserved_gb']:.3f} GB")
L.append(f"- Wall-clock training time: {r['train_wallclock_seconds']:.2f} s\n")

L.append("## Custom inference after save and reload\n")
for s, p in r["custom_inference_demo"].items():
    L.append(f"- \"{s}\" -> {p}")

open(f"{out_dir}/SUMMARY.md", "w").write("\n".join(L) + "\n")
print(f"wrote {out_dir}/SUMMARY.md")
