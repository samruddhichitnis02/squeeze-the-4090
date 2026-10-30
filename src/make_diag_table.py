"""Table for the Step 7a bottleneck diagnostic: API-server processes x client processes x streaming."""
import glob
import json
import os
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/step7/diag"
NAMES = {
    "asc1_procs1": "1 API server, 1 client process, streaming",
    "asc1_procs4": "1 API server, 4 client processes, streaming",
    "asc4_procs1": "4 API servers, 1 client process, streaming",
    "asc4_procs4": "4 API servers, 4 client processes, streaming",
    "asc4_stream": "4 API servers, 2 client processes, streaming",
    "asc4_nostream": "4 API servers, 2 client processes, non-streaming",
}
rows = {}
for p in glob.glob(f"{out_dir}/*.json"):
    key = os.path.basename(p)[:-5]
    rows[key] = {r["offered_rate"]: r for r in json.load(open(p))["rates"]}
rates = sorted({x for v in rows.values() for x in v})
md = ["# Step 7a: what limited the Step 6 plateau? (bf16, short prompts, output tokens/sec achieved)", "",
      "| Setup | " + " | ".join(f"offered {x:g} req/s" for x in rates) + " |", "|---|" + "---|" * len(rates)]
for key in NAMES:
    if key not in rows:
        continue
    md.append(f"| {NAMES[key]} | " + " | ".join(
        f"{rows[key][x]['output_tokens_per_sec']:.0f}" if x in rows[key] else "-" for x in rates) + " |")
md += ["", "Reference points: Step 6 (1 API server, 1 client) plateau about 18,000 tok/s; engine-only (no HTTP) at 256 concurrent "
       "sequences 28,754 tok/s (`results/step6/engine_only/`).", ""]
open(f"{out_dir}/diag_table.md", "w").write("\n".join(md))
print("\n".join(md))
