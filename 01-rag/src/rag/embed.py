import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
DIM = 768  # pgvector indexa hasta 2.000 dimensiones; 768 da buen balance


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
            for attempt in range(5):
                try:
                    r = self.client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task, output_dimensionality=DIM),
                    )
                    out.extend(e.values for e in r.embeddings)
                    print(f"  gemini: {len(out)}/{len(texts)}", flush=True)
                    break
                except errors.APIError as e:
                    if e.code == 429 and attempt < 7:  # límite del tier gratuito: espera y reintenta
                        print(f"  429, espero {min(60, 5 * 2**attempt)}s...", flush=True)
                        time.sleep(min(60, 5 * 2**attempt))
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
        return self.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, show_progress_bar=True)

    def query(self, text: str) -> np.ndarray:
        return self.model.encode(f"query: {text}", normalize_embeddings=True)



def load_chunks(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def embed_cached(embedder, chunks: list[dict], chunks_path: str) -> np.ndarray:
    cache = Path(f"data/emb_{embedder.name}_{Path(chunks_path).stem}.npy")
    if cache.exists():
        return np.load(cache)
    matrix = embedder.docs([c["text"] for c in chunks])
    np.save(cache, matrix)
    return matrix


def hit_at_5(embedder, matrix: np.ndarray, chunks: list[dict], qs: pd.DataFrame) -> int:
    hits = 0
    for q in qs.itertuples():
        scores = matrix @ embedder.query(q.question)  # coseno = producto punto con vectores normalizados
        top = np.argsort(-scores)[:5]
        hits += any(chunks[i]["source"] == q.source and str(chunks[i]["page"]) == q.page for i in top)
    return hits


if __name__ == "__main__":
    chunks_path = sys.argv[1]  # ej. data/chunks_fixed_400_50.jsonl
    chunks = load_chunks(chunks_path)
    qs = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
    qs = qs[qs.expected != "NO_SE"]
    for embedder in (GeminiEmbedder(), LocalEmbedder()):
        matrix = embed_cached(embedder, chunks, chunks_path)
        print(f"{embedder.name}: {hit_at_5(embedder, matrix, chunks, qs)}/{len(qs)} preguntas con la página correcta en el top-5")