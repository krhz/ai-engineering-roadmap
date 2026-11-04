import json
import sys

import pandas as pd

path = sys.argv[1]  # ej. data/chunks_fixed_400_50.jsonl
chunks = [json.loads(line) for line in open(path, encoding="utf-8")]
qs = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
qs = qs[qs.expected != "NO_SE"]

found = 0
for q in qs.itertuples():
    hits = [c for c in chunks if q.expected.lower() in c["text"].lower()]
    ok = any(c["source"] == q.source and str(c["page"]) == q.page for c in hits)
    found += ok
    print(f"{q.id}: {'OK ' if ok else 'NO '} ({len(hits)} chunks contienen '{q.expected}')")
print(f"Cobertura: {found}/{len(qs)} preguntas tienen su respuesta completa dentro de un chunk")