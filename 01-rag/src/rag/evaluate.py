import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from rag.generate import InvalidCitation, build_context, llm, rag


def plain(text: str) -> str:
    """Minúsculas y sin tildes, para comparar sin depender de acentos."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def norm(text: str) -> str:
    """plain() + quita los separadores de miles: "$18.200.000", "18,200,000" y "18200000" quedan iguales."""
    return re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", plain(text))


# ---------- 1. Correr el RAG sobre todo el set (con caché) ----------
def run_all(qs: pd.DataFrame, label: str) -> list[dict]:
    path = Path(f"data/answers_{label}.jsonl")
    if path.exists():
        return [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = []
    for q in qs.itertuples():
        try:
            answer, hits = rag(q.question)
            row = {"id": q.id, "found": answer.found, "answer": answer.answer,
                   "cited": [[hits[n - 1][1], hits[n - 1][2]] for n in answer.citations],
                   "hits": [[h[1], h[2], h[3]] for h in hits], "error": None}
        except InvalidCitation as e:
            row = {"id": q.id, "found": False, "answer": "", "cited": [], "hits": [], "error": str(e)}
        rows.append(row)
        print(f"{q.id} listo")
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return rows


# ---------- 2. Revisión con regex (lo que propone el reel) ----------
def regex_pass(q, row: dict) -> bool:
    if q.expected == "NO_SE":
        return not row["found"]
    cited = bool(row["cited"])
    # (?<!\d) y (?!\d): "136972971" no debe coincidir dentro de "1369729710"
    correct = re.search(rf"(?<!\d){re.escape(norm(q.expected))}(?!\d)", norm(row["answer"])) is not None
    return row["found"] and cited and correct


# ---------- 3. Juez LLM: ¿cada afirmación sale del contexto? ----------
class Claim(BaseModel):
    claim: str
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim]


JUDGE = """Eres un evaluador estricto. Divide la RESPUESTA en afirmaciones atómicas (una idea cada una).
Para cada afirmación, supported=true solo si se deduce directamente de los FRAGMENTOS. No uses conocimiento propio."""


def groundedness(row: dict) -> float | None:
    if not row["found"] or not row["answer"]:
        return None  # "no lo sé" no se evalúa
    hits = [tuple([None] + h) for h in row["hits"]]  # (chunk_id, source, page, content)
    verdict = llm.chat.completions.parse(
        model=os.environ.get("JUDGE_MODEL", os.environ["CHAT_MODEL"]),
        messages=[{"role": "system", "content": JUDGE},
                  {"role": "user", "content": f"FRAGMENTOS:\n{build_context(hits)}\n\nRESPUESTA:\n{row['answer']}"}],
        response_format=Verdict,
        temperature=0,
    ).choices[0].message.parsed
    if not verdict.claims:
        return None
    return sum(c.supported for c in verdict.claims) / len(verdict.claims)


# ---------- 4. ¿Dónde falló? ----------
def failure_kind(q, row: dict) -> str:
    if row["error"]:
        return "cita inválida"
    if q.expected == "NO_SE":
        return "inventó una respuesta"
    retrieved = any(h[0] == q.source and str(h[1]) == q.page for h in row["hits"])
    if not retrieved:
        return "no recuperó la ficha"
    cited_right = any(c[0] == q.source and str(c[1]) == q.page for c in row["cited"])
    if not row["found"]:
        return "tenía el contexto y dijo no sé"
    return "citó otra ficha" if not cited_right else "respondió mal con buen contexto"


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "base"
    qs = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
    rows = {r["id"]: r for r in run_all(qs, label)}

    passed, failures, scores = 0, {}, []
    for q in qs.itertuples():
        row = rows[q.id]
        if regex_pass(q, row):
            passed += 1
        else:
            kind = failure_kind(q, row)
            failures[kind] = failures.get(kind, 0) + 1
            print(f"FALLA {q.id} ({kind}): {q.question}")
        g = groundedness(row)
        if g is not None:
            scores.append(g)

    print(f"\npass rate: {passed / len(qs):.0%}")
    print(f"groundedness promedio: {sum(scores) / max(len(scores), 1):.0%} (sobre {len(scores)} respuestas)")
    for kind, n in sorted(failures.items(), key=lambda x: -x[1]):
        print(f"  {kind}: {n}")