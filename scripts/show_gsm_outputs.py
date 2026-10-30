"""Print a few saved GSM8K generations: python scripts/show_gsm_outputs.py file.json.gz [n]"""
import gzip
import json
import sys

texts = json.load(gzip.open(sys.argv[1], "rt", encoding="utf-8"))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
for i, t in enumerate(texts[:n]):
    print(f"--- output {i} ({len(t)} chars) ---")
    print(t[:350].replace("\n", " | "))
    print("   ...", t[-120:].replace("\n", " | "))
