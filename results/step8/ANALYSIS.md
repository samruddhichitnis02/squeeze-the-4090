# Step 8: quantisation, speed and measured quality (RTX 4090, Qwen2.5-1.5B-Instruct, vLLM 0.32)

Data: `quant_table.md` / `quant_table.csv` (full tables), `quant_overview.png`, per-configuration raw results and GSM8K outputs in `raw/`,
simulations in `kv_range.json`, `kv_fakequant.json`, `kv_fakequant_channel.json`, logs in `logs/step8/`.

Quality sets: WikiText-2 test perplexity (298,646 tokens, 1,024-token windows), MMLU zero-shot (all 14,042 questions, answer = arg-max over the four letter tokens),
GSM8K zero-shot chain-of-thought (all 1,319 problems, greedy, 512 tokens max, last "answer is N" number). Every configuration ran in its own vLLM process; speed was
measured in the same process. Differences are **paired against the bf16 reference per question**, with a standard error (SE) from the per-question differences.
Everything below is one run per configuration except bf16, which was run twice to measure run-to-run variation.

## Summary

| Configuration | Weights (GiB) | KV capacity (tokens) | Batch-1 ms/token | Perplexity | MMLU | GSM8K | Verdict |
|---|---|---|---|---|---|---|---|
| bf16 (reference) | 2.98 | 384,208 | 4.57 | 10.71 | 58.45% | 74.45% | reference |
| fp16 | 2.98 | 383,200 | 4.60 | 10.70 | 58.29% | 75.13% | same as bf16 within noise |
| FP8 weights | 1.73 | 422,912 | 3.91 (0.86x) | 10.81 | 58.24% | 75.13% | accuracy unchanged within test resolution; perplexity +1.0% |
| INT4 AWQ (weight-only) | 1.10 | 454,944 | 2.53 (0.55x) | 11.62 | 55.36% | 69.14% | clear but moderate loss: -3.1 MMLU, -5.3 GSM8K pts |
| INT4 GPTQ (weight-only) | 1.10 | 453,856 | 2.54 (0.55x) | 11.97 | 55.93% | 67.78% | clear but moderate loss: -2.5 MMLU, -6.7 GSM8K pts |
| FP8 KV cache, e4m3 fixed scale (Triton) | 2.98 | 768,736 | 4.25 (control 4.24) | 30.42 | 52.44% | 3.11% | **broken** |
| FP8 KV cache, per-token-head scale (Triton) | 2.98 | 742,896 | 4.31 | 49.78 | 41.11% | 1.52% | **broken** |
| FP8 weights + FP8 KV (Triton) | 1.73 | 845,248 | 3.68 | 30.96 | 51.96% | 4.32% | **broken** |
| FP8 KV, e5m2 | n/a | n/a | n/a | n/a | n/a | n/a | failed to start (vLLM torch.compile "data-dependent assertion" error) |

## 1. FP8 weights: safe for accuracy at this resolution, modest speedup at batch 1
- **Accuracy:** MMLU -0.21 pts (paired SE 0.18), GSM8K +0.68 pts (SE 0.96). Both inside noise; the tests could have detected a loss of about 0.4 MMLU points or 2 GSM8K points. The bf16-versus-bf16
  repeat differed by -0.23 GSM8K points (SE 0.67), so differences of that size are noise.
- **Perplexity does move:** 10.81 vs 10.71, +1.0%, while fp16 is -0.1%. Perplexity is a more sensitive instrument than the accuracy tests, so this is a small but probably real degradation.
- **Speed:** batch-1 time per token 3.91 ms vs 4.57 (0.86x, 14% faster), consistent with the Step 6 low-load finding. At 256 sequences throughput is unchanged (28,094 vs 28,392 tokens/s).
- **Memory:** weights 1.73 vs 2.98 GiB; 10.1% more KV capacity.
- GSM8K answers are word-for-word identical to bf16 for only 3.6% of problems, versus 46% for the bf16 repeat. The numerics clearly differ even though the accuracy does not.

## 2. INT4 weight-only: the biggest speed-up, with a real but moderate quality cost
- Batch-1 time per token fell from 4.57 to 2.53 ms (0.55x), the largest speed-up in the project, and weights shrink to 1.10 GiB. This is not 4-bit arithmetic: the weights are stored in 4 bits and
  expanded to 16-bit inside the kernel. The speed-up comes from reading fewer bytes per step, and it fits the Step 7 roofline picture (decode at small batch is limited by weight reads).
- Quality: MMLU -3.1 (AWQ) and -2.5 (GPTQ) points, GSM8K -5.3 and -6.7 points, perplexity +8.5% and +11.8%. All differences are far outside noise (4.7 to 10 SE).
  For context, Qwen's published table for the *older* Qwen2-1.5B-Instruct shows INT4 costing 0.8 to 1.7 MMLU points; this model and test set differ, so it is a reference point, not a validation. The loss here is larger.
- At 256 concurrent sequences, INT4 is only 3 to 6% faster than bf16 (30,014 / 29,376 vs 28,392 tokens/s). The benefit is concentrated at low batch, where decode is bandwidth bound.
- The two INT4 methods are close. AWQ has the better perplexity (11.62 vs 11.97) and GSM8K (69.1% vs 67.8%); GPTQ has the better MMLU (55.9% vs 55.4%). With single runs and small gaps, I do not rank them.

## 3. FP8 KV cache: speed and capacity gains, but model quality collapses on this model
- **Capacity doubles** (384k to 769k tokens), and speed is unchanged (4.25 vs 4.24 ms per token against the matching Triton bf16 control). Batch-256 throughput is 4% to 5% higher, within the noise seen across runs.
- **Quality collapses:** GSM8K falls from 73.9% (Triton bf16 control) to 3.1%, perplexity triples from 10.71 to 30.4, MMLU -6.0 points. The generated text shows repeated words ("total total total"),
  a runaway "'s's's's" loop in one answer, and numbers that do not follow from the steps. Combining it with FP8 weights gives the same result (4.3% GSM8K), so the damage comes from the KV cache.
  Zero of 1,319 GSM8K answers match bf16 word for word.
- **This is not a vLLM or Triton bug that I can find.** A plain-PyTorch simulation (quantise keys and values to FP8 and back before attention reads them, no vLLM) reproduces the failure:
  perplexity 26.1 (E4M3 fixed scale) and 46.4 (per-token-head scale) against 10.85 unquantised, matching vLLM's 30.4 and 49.8 closely. The same simulation shows the cause is in the **keys**:
  quantising only the values leaves perplexity at 10.86, quantising only the keys gives 26.1, the same as both.
- **What I ruled out and what I could not explain:**
  - Range overflow: no key or value exceeds the E4M3 limit of 448 (largest key 322 in layer 0, largest value 29), so nothing clips.
  - Dynamic per-token-head scaling is *worse* than a fixed scale (46.4 vs 26.1 simulated), the opposite of what dynamic scaling is meant to do.
  - Outlier key channels with a static per-channel scale (calibrated on separate text) did not help: perplexity 28.8. Early layers have a wide channel-range spread (65x in layer 0), yet fixing the scale per channel did not repair it.
  - **E5M2** is much worse in simulation (127 to 326) since it has fewer mantissa bits, and vLLM could not start it at all.
  - So the failure is a precision problem on the keys at 3 mantissa bits for this model, and I do not have a confirmed mechanism. I did not test other schemes (for example per-channel scales fitted per layer on more data, higher-precision keys with FP8 values, or hybrid layouts).
- **Practical reading:** on Qwen2.5-1.5B-Instruct in vLLM 0.32, do not enable an FP8 KV cache. The uncalibrated default is unsafe, and the alternatives I tried did not fix it. FP8 values with 16-bit keys looks harmless in simulation (perplexity 10.86), but no serving engine option for that exists in this build, so it was not measured end-to-end.

## 4. Run-to-run noise
- Two identical bf16 runs agree on MMLU to 0.01 points, on GSM8K to 0.23 points (SE 0.67), and on perplexity exactly, but their GSM8K outputs match word for word on only **46.4%** of problems.
  vLLM's greedy decoding is therefore not exactly reproducible across runs (likely different batching changing floating-point summation order), which is why "identical outputs" is not a usable test on its own.
- fp16 and Triton-bf16 land inside the same noise (GSM8K +0.68 and -0.53 points). Batch-1 speed repeats to 0.4% (4.57 vs 4.59 ms); batch-256 throughput to 0.3%.
- Speed comparisons across attention backends are confounded: the Triton bf16 control is 7% faster per token than the default FlashAttention bf16 (4.24 vs 4.57 ms), so FP8-KV and Triton rows must be compared with the Triton control, not the default.

## 5. Caveats
- One model (Qwen2.5-1.5B-Instruct). The FP8-KV failure and the INT4 loss may differ for other models, and INT4 weight-only usually looks better on larger models. A 7B model does not fit in the memory budget used here.
- One run per configuration except bf16. MMLU and GSM8K are single prompt formats; GSM8K scoring uses the last number or an "answer is" phrase and could mis-score a few answers in every configuration equally.
- The reference scores (58.45% MMLU, 74.45% GSM8K) are my own measurements. I searched for official figures for this exact model and found none, so they are not validated against a published number. (During the run I wrongly said the pipeline "reproduces known results"; that was not supported.) The prompt formats differ from the model card's, so absolute scores are not comparable to published ones; only the differences between configurations, measured with identical prompts, are meaningful.
- The batch-1 speed numbers are from 12 calls per configuration; the batch-256 numbers are engine-only (no HTTP stack), so they exceed the online results in Steps 6 and 7.
- The e5m2 KV configuration crashed at startup and is not in the quality tables.
- Datasets are downloaded to `data/` (git-ignored) by `src/prep_eval_data.py`; they are not redistributed here.
