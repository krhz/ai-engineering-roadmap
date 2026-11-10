import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
DIM = 768  # dimensiones que pedimos a Gemini; e5-small siempre da 384


def normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


class GeminiEmbedder:
    name = "gemini"

    def __init__(self) -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 50):  # lotes de 50 textos
            batch = texts[i : i + 50]
            for attempt in range(8):
                try:
                    r = self.client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task, output_dimensionality=DIM),
                    )
                    out.extend(e.values for e in r.embeddings)
                    break
                except errors.APIError as e:
                    if e.code == 429 and attempt < 7:  # límite por minuto del tier gratuito
                        wait = min(60, 5 * 2**attempt)
                        print(f"  429, espero {wait}s...", flush=True)
                        time.sleep(wait)
                        continue
                    raise
        return normalize(np.array(out, dtype=np.float32))

    def docs(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class LocalEmbedder:
    name = "e5"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("intfloat/multilingual-e5-small")  # 384 dimensiones

    def docs(self, texts: list[str]) -> np.ndarray:
        # e5 exige el prefijo "passage: " en documentos y "query: " en preguntas
        return self.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True)

    def query(self, text: str) -> np.ndarray:
        return self.model.encode(f"query: {text}", normalize_embeddings=True)


def load_chunks(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def embed_cached(embedder, chunks: list[dict], chunks_path: str, tag: str = "", block: int = 200) -> np.ndarray:
    cache = Path(f"data/emb_{embedder.name}_{Path(chunks_path).stem}{tag}.npy")
    if cache.exists():
        return np.load(cache)
    partial = Path(str(cache).replace(".npy", ".partial.npy"))
    done = np.load(partial) if partial.exists() else None  # retoma si se cayó antes
    texts = [c["text"] for c in chunks]
    for i in range(0 if done is None else len(done), len(texts), block):
        part = embedder.docs(texts[i : i + block])
        done = part if done is None else np.vstack([done, part])
        np.save(partial, done)  # guarda el avance cada `block` chunks
        print(f"  {embedder.name}: {len(done)}/{len(texts)}", flush=True)
    np.save(cache, done)
    partial.unlink()
    return done


def subset(chunks: list[dict], qs: pd.DataFrame, window: int = 5) -> list[dict]:
    """Docs cortos completos + páginas cerca de cada respuesta. Para comparar con Gemini sin agotar la cuota."""
    gold: dict[str, list[int]] = {}
    for q in qs.itertuples():
        gold.setdefault(q.source, []).append(int(q.page))
    last: dict[str, int] = {}
    for c in chunks:
        last[c["source"]] = max(last.get(c["source"], 0), c["page"])
    return [c for c in chunks
            if last[c["source"]] <= 20 or any(abs(c["page"] - p) <= window for p in gold.get(c["source"], []))]


def hit_at_5(embedder, matrix: np.ndarray, chunks: list[dict], qs: pd.DataFrame) -> int:
    hits = 0
    for q in qs.itertuples():
        scores = matrix @ embedder.query(q.question)  # coseno = producto punto con vectores normalizados
        top = np.argsort(-scores)[:5]
        hits += any(chunks[i]["source"] == q.source and str(chunks[i]["page"]) == q.page for i in top)
    return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks_path")  # ej. data/chunks_fixed_400_50.jsonl
    ap.add_argument("--models", default="e5", help="e5, gemini o e5,gemini")
    ap.add_argument("--subset", action="store_true", help="solo páginas cerca de las respuestas")
    a = ap.parse_args()

    chunks = load_chunks(a.chunks_path)
    qs = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
    qs = qs[qs.expected != "NO_SE"]
    tag = "_subset" if a.subset else ""
    if a.subset:
        chunks = subset(chunks, qs)
    print(f"{len(chunks)} chunks")
    for name in a.models.split(","):
        embedder = GeminiEmbedder() if name == "gemini" else LocalEmbedder()
        matrix = embed_cached(embedder, chunks, a.chunks_path, tag)
        print(f"{embedder.name}: {hit_at_5(embedder, matrix, chunks, qs)}/{len(qs)} preguntas con la página correcta en el top-5")