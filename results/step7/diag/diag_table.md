# Step 7a: what limited the Step 6 plateau? (bf16, short prompts, output tokens/sec achieved)

| Setup | offered 192 req/s | offered 256 req/s | offered 384 req/s | offered 512 req/s |
|---|---|---|---|---|
| 1 API server, 1 client process, streaming | 19285 | 19414 | 18689 | - |
| 1 API server, 4 client processes, streaming | 14470 | 16229 | 15786 | - |
| 4 API servers, 1 client process, streaming | 19402 | 19953 | 21472 | - |
| 4 API servers, 4 client processes, streaming | 18500 | 19571 | 20349 | - |
| 4 API servers, 2 client processes, streaming | - | 19561 | 21361 | 21864 |
| 4 API servers, 2 client processes, non-streaming | - | 20148 | 21366 | 21943 |

Reference points: Step 6 (1 API server, 1 client) plateau about 18,000 tok/s; engine-only (no HTTP) at 256 concurrent sequences 28,754 tok/s (`results/step6/engine_only/`).
