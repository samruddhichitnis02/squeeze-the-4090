# Step 7: serving knob sweep (RTX 4090, Qwen2.5-1.5B-Instruct, vLLM 0.32)

Every configuration: bf16 unless stated, 4 API server processes, 2 client processes, max-model-len 2048, gpu-memory-utilization 0.6, 128 output tokens per request, streamed, open-loop Poisson arrivals, one run per cell.
Workloads: **short** = short chat prompts (about 41 tokens, decode dominated); **longprefix** = a ~950-token document identical for every request plus a short unique question (prefill and KV-read heavy).
SLA: TTFT <= 500 ms and TPOT <= 50 ms. SLA capacity = highest tested rate with >= 95% of requests inside the SLA (limited by the small rate grid). Single-run differences under roughly 15 to 20% are within noise (Step 6).

## Workload: short chat prompts

| Configuration | KV cache (tokens) | SLA capacity (req/s) | Peak tok/s | e2e p99 @96 req/s (ms) | e2e p99 @160 req/s (ms) | TTFT p99 @96 (ms) | TTFT p99 @160 (ms) | TPOT p50 @96 (ms) |
|---|---|---|---|---|---|---|---|---|
| base (bf16, prefix cache on, chunked prefill on, 256 seqs) | 397968 | 96 | 19166 | 1039 | 2595 | 155 | 1093 | 6.7 |
| prefix caching off | 397968 | 96 | 17553 | 930 | 3644 | 34 | 2145 | 6.6 |
| chunked prefill off | 397968 | 96 | 19088 | 926 | 2568 | 33 | 1064 | 6.7 |
| max-num-batched-tokens 512 | 386448 | 96 | 19333 | 1004 | 2599 | 40 | 1114 | 7.0 |
| max-num-seqs 32 | 399728 | 32 | 5987 | 15204 | 37092 | 14529 | 36426 | 5.2 |
| max-num-seqs 64 | 397744 | 32 | 10534 | 3176 | 15149 | 2432 | 14362 | 6.0 |
| FP8 weights | 425136 | 96 | 17322 | 1070 | 5114 | 38 | 3602 | 7.7 |
| [Triton attention] bf16 weights, bf16 KV (control) | 386048 | 160 | 21950 | 999 | 1504 | 40 | 55 | 6.6 |
| [Triton attention] bf16 weights, FP8 KV cache | 772400 | 160 | 22131 | 1145 | 1392 | 47 | 49 | 7.2 |
| [Triton attention] FP8 weights, bf16 KV (control) | 425200 | 96 | 18685 | 1638 | 4822 | 57 | 3147 | 11.4 |
| [Triton attention] FP8 weights + FP8 KV cache | 851872 | 96 | 21113 | 1270 | 2101 | 47 | 578 | 8.4 |

## Workload: long shared prefix (~950 prompt tokens)

| Configuration | KV cache (tokens) | SLA capacity (req/s) | Peak tok/s | e2e p99 @32 req/s (ms) | e2e p99 @64 req/s (ms) | TTFT p99 @32 (ms) | TTFT p99 @64 (ms) | TPOT p50 @32 (ms) |
|---|---|---|---|---|---|---|---|---|
| base (bf16, prefix cache on, chunked prefill on, 256 seqs) | 397968 | 64 (>= , ladder top) | 7262 | 866 | 1687 | 29 | 56 | 6.0 |
| prefix caching off | 397968 | 16 | 3110 | 10059 | 31159 | 3141 | 26383 | 48.1 |
| chunked prefill off | 397968 | 64 (>= , ladder top) | 7262 | 875 | 1523 | 30 | 51 | 6.0 |
| max-num-batched-tokens 512 | 386448 | 64 (>= , ladder top) | 7269 | 864 | 1645 | 30 | 53 | 6.0 |
| [prefix cache off] chunked prefill off | 397968 | 16 | 3071 | 8229 | 29700 | 3820 | 26241 | 38.7 |
| [prefix cache off] max-num-batched-tokens 512 | 399616 | 16 | 3210 | 5663 | 23262 | 3589 | 21275 | 18.7 |
| max-num-seqs 32 | 399728 | 32 | 4962 | 976 | 9279 | 175 | 8491 | 6.0 |
| max-num-seqs 64 | 397744 | 32 | 7236 | 880 | 2282 | 29 | 1211 | 6.0 |
| FP8 weights | 425136 | 64 (>= , ladder top) | 7327 | 663 | 1810 | 27 | 61 | 4.7 |
| [Triton attention] bf16 weights, bf16 KV (control) | 386048 | 64 (>= , ladder top) | 7287 | 744 | 901 | 31 | 39 | 5.3 |
| [Triton attention] bf16 weights, FP8 KV cache | 772400 | 64 (>= , ladder top) | 7289 | 744 | 899 | 35 | 40 | 5.3 |
| [Triton attention] FP8 weights, bf16 KV (control) | 425200 | 64 (>= , ladder top) | 7347 | 832 | 1356 | 49 | 56 | 5.2 |
| [Triton attention] FP8 weights + FP8 KV cache | 851872 | 64 (>= , ladder top) | 7354 | 735 | 1017 | 37 | 47 | 4.6 |

## Full per-rate results

### base (bf16, prefix cache on, chunked prefill on, 256 seqs) / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 29.9 | 3824 | 691 / 706 | 17 / 22 | 5.3 / 5.4 | 29.9 |
| 96 | 87.6 | 11208 | 876 / 1039 | 23 / 155 | 6.7 / 8.0 | 87.6 |
| 160 | 142.4 | 18221 | 1926 / 2595 | 240 / 1093 | 13.4 / 13.4 | 87.5 |
| 256 | 149.7 | 19166 | 4263 / 6876 | 2625 / 5714 | 12.9 / 13.0 | 23.3 |

### prefix caching off / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 29.9 | 3826 | 690 / 748 | 18 / 34 | 5.3 / 5.7 | 29.9 |
| 96 | 87.7 | 11221 | 861 / 930 | 22 / 34 | 6.6 / 7.1 | 87.7 |
| 160 | 133.9 | 17141 | 2484 / 3644 | 706 / 2145 | 14.1 / 14.9 | 50.4 |
| 256 | 137.1 | 17553 | 5080 / 8345 | 3288 / 7159 | 14.1 / 14.2 | 18.5 |

### chunked prefill off / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 29.9 | 3823 | 695 / 716 | 17 / 26 | 5.3 / 5.5 | 29.9 |
| 96 | 87.5 | 11205 | 875 / 926 | 22 / 33 | 6.7 / 7.1 | 87.5 |
| 160 | 142.6 | 18251 | 1907 / 2568 | 223 / 1064 | 13.3 / 13.4 | 88.3 |
| 256 | 149.1 | 19088 | 4306 / 6944 | 2656 / 5779 | 12.9 / 13.0 | 22.6 |

### max-num-batched-tokens 512 / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 29.9 | 3824 | 707 / 744 | 18 / 34 | 5.4 / 5.7 | 29.9 |
| 96 | 87.5 | 11206 | 910 / 1004 | 24 / 40 | 7.0 / 7.7 | 87.5 |
| 160 | 142.3 | 18213 | 1943 / 2599 | 258 / 1114 | 13.3 / 13.4 | 87.1 |
| 256 | 151.0 | 19333 | 4185 / 6735 | 2572 / 5586 | 12.7 / 12.9 | 22.8 |

### max-num-seqs 32 / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 29.9 | 3829 | 674 / 779 | 17 / 102 | 5.2 / 5.6 | 29.9 |
| 96 | 46.8 | 5987 | 7781 / 15204 | 7125 / 14529 | 5.2 / 5.6 | 2.1 |
| 160 | 45.8 | 5858 | 19007 / 37092 | 18307 / 36426 | 5.4 / 5.7 | 0.7 |
| 256 | 45.9 | 5875 | 21911 / 42829 | 21231 / 42170 | 5.4 / 5.6 | 0.6 |

### max-num-seqs 64 / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 29.9 | 3827 | 675 / 712 | 17 / 26 | 5.2 / 5.5 | 29.9 |
| 96 | 78.6 | 10063 | 1932 / 3176 | 1188 / 2432 | 6.0 / 6.7 | 14.2 |
| 160 | 79.2 | 10135 | 7709 / 15149 | 6925 / 14362 | 6.1 / 6.7 | 4.2 |
| 256 | 82.3 | 10534 | 9977 / 19709 | 9225 / 19046 | 5.9 / 6.2 | 2.2 |

### FP8 weights / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 30.2 | 3862 | 586 / 629 | 16 / 30 | 4.5 / 4.8 | 30.2 |
| 96 | 88.2 | 11291 | 1002 / 1070 | 25 / 38 | 7.7 / 8.2 | 88.2 |
| 160 | 123.3 | 15778 | 3244 / 5114 | 1229 / 3602 | 15.7 / 15.9 | 35.7 |
| 256 | 135.3 | 17322 | 5169 / 8606 | 3344 / 7383 | 14.4 / 14.4 | 18.6 |

### [Triton attention] bf16 weights, bf16 KV (control) / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 30.0 | 3836 | 663 / 728 | 17 / 33 | 5.1 / 5.6 | 30.0 |
| 96 | 87.8 | 11240 | 865 / 999 | 24 / 40 | 6.6 / 7.6 | 87.8 |
| 160 | 148.8 | 19050 | 1232 / 1504 | 34 / 55 | 9.4 / 11.5 | 148.8 |
| 256 | 171.5 | 21950 | 3364 / 4924 | 1942 / 3915 | 11.1 / 12.2 | 36.6 |

### [Triton attention] bf16 weights, FP8 KV cache / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 30.0 | 3835 | 654 / 710 | 17 / 32 | 5.0 / 5.4 | 30.0 |
| 96 | 87.8 | 11239 | 943 / 1145 | 27 / 47 | 7.2 / 8.8 | 87.8 |
| 160 | 149.3 | 19116 | 1311 / 1392 | 35 / 49 | 10.0 / 10.7 | 149.3 |
| 256 | 172.9 | 22131 | 3232 / 4823 | 1756 / 3788 | 11.0 / 11.6 | 36.9 |

### [Triton attention] FP8 weights, bf16 KV (control) / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 30.3 | 3873 | 667 / 859 | 25 / 44 | 5.0 / 6.5 | 30.3 |
| 96 | 88.2 | 11284 | 1490 / 1638 | 41 / 57 | 11.4 / 12.6 | 88.2 |
| 160 | 125.8 | 16104 | 3237 / 4822 | 1337 / 3147 | 15.0 / 15.7 | 26.4 |
| 256 | 146.0 | 18685 | 4543 / 7375 | 2847 / 6312 | 13.0 / 13.5 | 15.6 |

### [Triton attention] FP8 weights + FP8 KV cache / short

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 32 | 30.3 | 3876 | 589 / 939 | 19 / 186 | 4.5 / 7.2 | 30.3 |
| 96 | 88.4 | 11318 | 1093 / 1270 | 31 / 47 | 8.4 / 9.8 | 88.4 |
| 160 | 147.3 | 18855 | 1805 / 2101 | 97 / 578 | 12.3 / 14.0 | 135.4 |
| 256 | 164.9 | 21113 | 3519 / 5470 | 2057 / 4450 | 11.5 / 11.9 | 30.5 |

### base (bf16, prefix cache on, chunked prefill on, 256 seqs) / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 943 | 704 / 721 | 18 / 22 | 5.4 / 5.5 | 7.4 |
| 16 | 14.4 | 1841 | 689 / 736 | 19 / 26 | 5.3 / 5.6 | 14.4 |
| 32 | 30.0 | 3841 | 784 / 866 | 22 / 29 | 6.0 / 6.6 | 30.0 |
| 64 | 56.7 | 7262 | 1343 / 1687 | 36 / 56 | 10.3 / 13.0 | 56.7 |

### prefix caching off / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 944 | 821 / 1086 | 35 / 57 | 6.2 / 8.3 | 7.4 |
| 16 | 14.4 | 1837 | 915 / 1159 | 37 / 89 | 6.9 / 8.8 | 14.4 |
| 32 | 24.3 | 3110 | 7114 / 10059 | 211 / 3141 | 48.1 / 77.3 | 6.4 |
| 64 | 21.8 | 2795 | 23669 / 31159 | 11997 / 26383 | 91.8 / 92.4 | 0.0 |

### chunked prefill off / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 943 | 707 / 723 | 19 / 25 | 5.4 / 5.5 | 7.4 |
| 16 | 14.4 | 1842 | 694 / 744 | 19 / 34 | 5.3 / 5.7 | 14.4 |
| 32 | 30.0 | 3842 | 788 / 875 | 22 / 30 | 6.0 / 6.7 | 30.0 |
| 64 | 56.7 | 7262 | 1315 / 1523 | 34 / 51 | 10.1 / 11.7 | 56.7 |

### max-num-batched-tokens 512 / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 943 | 706 / 722 | 19 / 23 | 5.4 / 5.5 | 7.4 |
| 16 | 14.4 | 1840 | 691 / 749 | 19 / 29 | 5.3 / 5.7 | 14.4 |
| 32 | 30.0 | 3843 | 787 / 864 | 22 / 30 | 6.0 / 6.6 | 30.0 |
| 64 | 56.8 | 7269 | 1346 / 1645 | 36 / 53 | 10.3 / 12.7 | 56.8 |

### [prefix cache off] chunked prefill off / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 943 | 806 / 1051 | 35 / 54 | 6.1 / 8.0 | 7.4 |
| 16 | 14.4 | 1840 | 934 / 1213 | 38 / 92 | 7.1 / 9.2 | 14.4 |
| 32 | 24.0 | 3071 | 6272 / 8229 | 172 / 3820 | 38.7 / 57.3 | 10.0 |
| 64 | 22.0 | 2821 | 19525 / 29700 | 12606 / 26241 | 54.6 / 60.0 | 0.0 |

### [prefix cache off] max-num-batched-tokens 512 / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 943 | 807 / 1046 | 39 / 64 | 6.0 / 7.9 | 7.4 |
| 16 | 14.4 | 1839 | 915 / 1173 | 42 / 127 | 6.9 / 8.9 | 14.4 |
| 32 | 24.6 | 3149 | 3643 / 5663 | 1262 / 3589 | 18.7 / 18.8 | 7.4 |
| 64 | 25.1 | 3210 | 13349 / 23262 | 10966 / 21275 | 18.8 / 19.1 | 1.0 |

### max-num-seqs 32 / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 944 | 705 / 721 | 19 / 25 | 5.4 / 5.5 | 7.4 |
| 16 | 14.4 | 1842 | 688 / 738 | 19 / 24 | 5.3 / 5.7 | 14.4 |
| 32 | 30.0 | 3843 | 787 / 976 | 22 / 175 | 6.0 / 6.3 | 30.0 |
| 64 | 38.8 | 4962 | 5217 / 9279 | 4425 / 8491 | 6.2 / 6.4 | 2.6 |

### max-num-seqs 64 / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 943 | 708 / 730 | 19 / 25 | 5.4 / 5.6 | 7.4 |
| 16 | 14.4 | 1841 | 695 / 738 | 20 / 32 | 5.3 / 5.7 | 14.4 |
| 32 | 30.0 | 3842 | 789 / 880 | 21 / 29 | 6.0 / 6.7 | 30.0 |
| 64 | 56.5 | 7236 | 1574 / 2282 | 508 / 1211 | 8.4 / 8.5 | 27.4 |

### FP8 weights / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 951 | 535 / 574 | 16 / 27 | 4.1 / 4.4 | 7.4 |
| 16 | 14.5 | 1862 | 527 / 574 | 16 / 27 | 4.0 / 4.4 | 14.5 |
| 32 | 30.4 | 3885 | 607 / 663 | 18 / 27 | 4.7 / 5.1 | 30.4 |
| 64 | 57.2 | 7327 | 1375 / 1810 | 36 / 61 | 10.5 / 13.9 | 57.2 |

### [Triton attention] bf16 weights, bf16 KV (control) / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 946 | 676 / 711 | 18 / 27 | 5.2 / 5.4 | 7.4 |
| 16 | 14.4 | 1846 | 655 / 687 | 18 / 30 | 5.0 / 5.3 | 14.4 |
| 32 | 30.1 | 3851 | 689 / 744 | 19 / 31 | 5.3 / 5.7 | 30.1 |
| 64 | 56.9 | 7287 | 810 / 901 | 24 / 39 | 6.2 / 6.9 | 56.9 |

### [Triton attention] bf16 weights, FP8 KV cache / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 945 | 662 / 686 | 18 / 26 | 5.1 / 5.2 | 7.4 |
| 16 | 14.4 | 1846 | 649 / 682 | 18 / 30 | 5.0 / 5.2 | 14.4 |
| 32 | 30.1 | 3852 | 697 / 744 | 20 / 35 | 5.3 / 5.7 | 30.1 |
| 64 | 56.9 | 7289 | 838 / 899 | 25 / 40 | 6.4 / 6.9 | 56.9 |

### [Triton attention] FP8 weights, bf16 KV (control) / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 954 | 525 / 622 | 21 / 32 | 4.0 / 4.7 | 7.4 |
| 16 | 14.5 | 1862 | 551 / 644 | 22 / 39 | 4.2 / 4.8 | 14.5 |
| 32 | 30.4 | 3887 | 695 / 832 | 27 / 49 | 5.2 / 6.3 | 30.4 |
| 64 | 57.4 | 7347 | 1202 / 1356 | 37 / 56 | 9.2 / 10.4 | 57.4 |

### [Triton attention] FP8 weights + FP8 KV cache / longprefix

| Offered (req/s) | Achieved (req/s) | Output tok/s | e2e p50 / p99 (ms) | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| 8 | 7.4 | 954 | 495 / 556 | 18 / 28 | 3.7 / 4.2 | 7.4 |
| 16 | 14.6 | 1865 | 521 / 605 | 19 / 37 | 3.9 / 4.6 | 14.6 |
| 32 | 30.4 | 3890 | 613 / 735 | 21 / 37 | 4.6 / 5.6 | 30.4 |
| 64 | 57.5 | 7354 | 882 / 1017 | 29 / 47 | 6.7 / 7.8 | 57.5 |

## Roofline cross-check

Ceilings used: memory bandwidth 1008 GB/s; bf16 tensor throughput 165.6 TFLOP/s dense with FP32 accumulate (stock RTX 4090, TechPowerUp measurement; the 82.6 TFLOPS quoted by some sites is the FP32 shader rate). Matmul FLOPs per token = 2 x parameters: 3.09 GFLOP per generated token, 2.62 GFLOP per prompt token (the output projection runs only on the last prompt position). Attention FLOPs are ignored (small at these context lengths).

### Decode at batch of about 1 is limited by weight reads

Uses the Step 6 load test at 2 req/s (about 1 request in flight, e2e about 0.5 to 0.6 s), not this sweep: the lowest rate here is 32 req/s, which keeps about 20 requests in flight. Observed TPOT p50 versus the time to read the weights once at full bandwidth.

| Configuration | Weight bytes read per step (GB, est.) | Bandwidth-bound step time (ms) | Observed TPOT p50 (ms) | Implied bandwidth use | Observed minus bound (ms) |
|---|---|---|---|---|---|
| bf16 weights (Step 6, 2 req/s) | 3.09 | 3.06 | 4.74 | 65% | 1.68 |
| FP8 weights (lm_head stays bf16) (Step 6, 2 req/s) | 1.78 | 1.76 | 3.82 | 46% | 2.06 |

FP8 reads 58% of the bf16 weight bytes, so a pure bandwidth model predicts TPOT at 58% of bf16; observed is 81%. A fixed cost per step of about 1.5 to 2 ms (kernel launches, attention, sampling, norms) that does not shrink with weight bytes would explain the gap. That reading is an inference from two data points, not a profile.

### Prefill throughput versus the compute ceiling (prefix caching off only)

TFLOP/s = generated tokens x decode FLOPs + prompt tokens x prefill FLOPs, from measured rates. Prompt FLOPs are only counted when prefix caching is off, because with caching the shared prefix is computed once and the true FLOPs are far lower (a no-cache assumption would give an invalid ~100% of peak). Percent is of 165.6 TFLOP/s. Valid only where the server was saturated (the rate ladder hit its overload stop).

| Workload | Configuration | Peak tok/s | Achieved req/s | Decode-only TFLOP/s | Decode + prefill TFLOP/s | % of bf16 peak | Saturated? |
|---|---|---|---|---|---|---|---|
| short | base (bf16, prefix cache on, chunked prefill on, 256 seqs) | 19166 | 149.7 | 59.2 | n/a (prefix cached or short prompts) | n/a | yes |
| short | prefix caching off | 17553 | 137.1 | 54.2 | 68.8 | 42% | yes |
| short | chunked prefill off | 19088 | 149.1 | 58.9 | n/a (prefix cached or short prompts) | n/a | yes |
| short | max-num-batched-tokens 512 | 19333 | 151.0 | 59.7 | n/a (prefix cached or short prompts) | n/a | yes |
| short | max-num-seqs 32 | 5987 | 46.8 | 18.5 | n/a (prefix cached or short prompts) | n/a | yes |
| short | max-num-seqs 64 | 10534 | 82.3 | 32.5 | n/a (prefix cached or short prompts) | n/a | yes |
| short | FP8 weights | 17322 | 135.3 | 53.5 | n/a (prefix cached or short prompts) | n/a | yes |
| short | [Triton attention] bf16 weights, bf16 KV (control) | 21950 | 171.5 | 67.8 | n/a (prefix cached or short prompts) | n/a | yes |
| short | [Triton attention] bf16 weights, FP8 KV cache | 22131 | 172.9 | 68.3 | n/a (prefix cached or short prompts) | n/a | yes |
| short | [Triton attention] FP8 weights, bf16 KV (control) | 18685 | 146.0 | 57.7 | n/a (prefix cached or short prompts) | n/a | yes |
| short | [Triton attention] FP8 weights + FP8 KV cache | 21113 | 164.9 | 65.2 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | base (bf16, prefix cache on, chunked prefill on, 256 seqs) | 7262 | 56.7 | 22.4 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | prefix caching off | 3110 | 24.3 | 9.6 | 70.2 | 42% | yes |
| longprefix | chunked prefill off | 7262 | 56.7 | 22.4 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | max-num-batched-tokens 512 | 7269 | 56.8 | 22.4 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | [prefix cache off] chunked prefill off | 3071 | 24.0 | 9.5 | 69.3 | 42% | yes |
| longprefix | [prefix cache off] max-num-batched-tokens 512 | 3210 | 25.1 | 9.9 | 72.4 | 44% | yes |
| longprefix | max-num-seqs 32 | 4962 | 38.8 | 15.3 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | max-num-seqs 64 | 7236 | 56.5 | 22.3 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | FP8 weights | 7327 | 57.2 | 22.6 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | [Triton attention] bf16 weights, bf16 KV (control) | 7287 | 56.9 | 22.5 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | [Triton attention] bf16 weights, FP8 KV cache | 7289 | 56.9 | 22.5 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | [Triton attention] FP8 weights, bf16 KV (control) | 7347 | 57.4 | 22.7 | n/a (prefix cached or short prompts) | n/a | yes |
| longprefix | [Triton attention] FP8 weights + FP8 KV cache | 7354 | 57.5 | 22.7 | n/a (prefix cached or short prompts) | n/a | yes |
