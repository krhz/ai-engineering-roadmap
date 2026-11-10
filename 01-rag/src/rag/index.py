import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.embed import LocalEmbedder, embed_cached, load_chunks

load_dotenv()

SCHEMA = f"""
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    chunk_id  text PRIMARY KEY,
    source    text NOT NULL,
    page      int  NOT NULL,
    content   text NOT NULL,
    embedding vector(384) NOT NULL,  -- e5-small da 384 dimensiones
    cargo        text,  -- metadatos de la ficha: sirven para filtrar antes de buscar
    departamento text,
    municipio    text,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (tsv);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # permite pasar arrays de numpy como vector
    return conn


def plain(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def load_meta(path: str = "my_docs/campanas_2023.csv") -> dict[int, dict]:
    """fila -> {cargo, departamento, municipio}, leído del CSV de fichas."""
    df = pd.read_csv(path, usecols=["fila", "cargo", "departamento", "municipio"]).fillna("")
    return {int(r.fila): {"cargo": r.cargo, "departamento": r.departamento, "municipio": r.municipio}
            for r in df.itertuples()}


def build_index(conn: psycopg.Connection, chunks: list[dict], matrix: np.ndarray) -> None:
    meta = load_meta()
    conn.execute(SCHEMA)
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (chunk_id, source, page, content, embedding, cargo, departamento, municipio)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            [(c["chunk_id"], c["source"], c["page"], c["text"], matrix[i],
              *meta.get(c["page"], {"cargo": None, "departamento": None, "municipio": None}).values())
             for i, c in enumerate(chunks)],
        )


_places: dict = {}


def detect_filters(conn, question: str) -> tuple[list[str], list[str]]:
    """Departamentos y municipios que la pregunta nombra (palabra completa, sin tildes)."""
    if not _places:
        for col in ("departamento", "municipio"):
            vals = [v for (v,) in conn.execute(f"SELECT DISTINCT {col} FROM chunks WHERE {col} <> ''")]
            _places[col] = [v for v in vals if len(v) >= 4]  # evita nombres cortos que salen en cualquier frase
    q = plain(question)
    found = lambda col: [v for v in _places[col] if re.search(rf"\b{re.escape(plain(v))}\b", q)]
    return found("departamento"), found("municipio")


def where_sql(filters) -> tuple[str, list]:
    if not filters or not (filters[0] or filters[1]):
        return "", []
    # OR: si "Nariño" es departamento y municipio, no descartamos ninguno de los dos
    return "WHERE (departamento = ANY(%s) OR municipio = ANY(%s))", [filters[0], filters[1]]


def vector_search(conn, qvec: np.ndarray, k: int = 20, filters=None) -> list[tuple]:
    where, params = where_sql(filters)
    return conn.execute(
        f"SELECT chunk_id, source, page, content FROM chunks {where} ORDER BY embedding <=> %s LIMIT %s",
        (*params, qvec, k),
    ).fetchall()


def keyword_search(conn, question: str, k: int = 20, filters=None) -> list[tuple]:
    words = " or ".join(re.findall(r"\w+", question))  # OR entre palabras: no exige que estén todas
    where, params = where_sql(filters)
    where = where.replace("WHERE", "AND")
    return conn.execute(
        f"""SELECT chunk_id, source, page, content
            FROM chunks, websearch_to_tsquery('spanish', %s) AS q
            WHERE tsv @@ q {where}
            ORDER BY ts_rank(tsv, q) DESC
            LIMIT %s""",
        (words, *params, k),
    ).fetchall()


def rrf(*result_lists: list[tuple], k: int = 60) -> list[tuple]:
    """Reciprocal Rank Fusion: suma 1/(k + posición) de cada lista."""
    scores: dict[str, float] = {}
    rows: dict[str, tuple] = {}
    for results in result_lists:
        for rank, row in enumerate(results):
            scores[row[0]] = scores.get(row[0], 0.0) + 1 / (k + rank + 1)
            rows[row[0]] = row
    return [rows[cid] for cid in sorted(scores, key=scores.get, reverse=True)]


class Reranker:
    def __init__(self, model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)  # multilingüe y liviano; más fuerte: BAAI/bge-reranker-v2-m3

    def rerank(self, question: str, rows: list[tuple], top: int = 5) -> list[tuple]:
        scores = self.model.predict([(question, r[3]) for r in rows])
        return [rows[i] for i in np.argsort(-scores)[:top]]


def search(conn, embedder, question: str, mode: str = "hybrid", k: int = 5, reranker: Reranker | None = None) -> list[tuple]:
    qvec = embedder.query(question)
    if mode == "vector":
        return vector_search(conn, qvec, k)
    filters = detect_filters(conn, question) if mode == "rerank_filtro" else None
    candidates = rrf(vector_search(conn, qvec, filters=filters), keyword_search(conn, question, filters=filters))
    if mode == "hybrid":
        return candidates[:k]
    return reranker.rerank(question, candidates[:20], top=k)  # mode == "rerank" o "rerank_filtro"


def evaluate_retrieval(conn, embedder, qs: pd.DataFrame, mode: str, reranker=None) -> tuple[float, float]:
    hits, rr = 0, 0.0
    for q in qs.itertuples():
        results = search(conn, embedder, q.question, mode, k=5, reranker=reranker)
        ranks = [i for i, r in enumerate(results, start=1) if r[1] == q.source and str(r[2]) == q.page]
        if ranks:
            hits += 1
            rr += 1 / ranks[0]
    return hits / len(qs), rr / len(qs)


if __name__ == "__main__":
    chunks_path = sys.argv[1]  # ej. data/chunks_fixed_400_50.jsonl
    chunks = load_chunks(chunks_path)
    embedder = LocalEmbedder()  # el mismo modelo que embebió el corpus
    conn = connect()
    build_index(conn, chunks, embed_cached(embedder, chunks, chunks_path))
    print(f"Indexados {len(chunks)} chunks")

    qs = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
    qs = qs[qs.expected != "NO_SE"]
    reranker = Reranker()
    for mode in ("vector", "hybrid", "rerank", "rerank_filtro"):
        recall, mrr = evaluate_retrieval(conn, embedder, qs, mode, reranker)
        print(f"{mode:13}  recall@5={recall:.2f}  MRR={mrr:.2f}")