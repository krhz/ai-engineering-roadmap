import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken

from rag.load import Page, load

ENC = tiktoken.get_encoding("cl100k_base")  # aproximado: cada modelo tiene su tokenizer


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


@dataclass
class Chunk:
    chunk_id: str  # ej. "campanas_2023-p812-c0" (p = fila de la ficha)
    source: str
    page: int
    text: str
    tokens: int


def chunk_fixed(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    """Ventana fija de `size` tokens que avanza `size - overlap` tokens cada vez."""
    chunks: list[Chunk] = []
    step = size - overlap
    for p in pages:
        ids = ENC.encode(p.text)
        for i, start in enumerate(range(0, max(len(ids), 1), step)):
            window = ids[start : start + size]
            if not window:
                break
            text = ENC.decode(window)
            chunks.append(Chunk(f"{p.doc_id}-p{p.page}-c{i}", p.source, p.page, text, len(window)))
            if start + size >= len(ids):
                break
    return chunks


def chunk_by_paragraph(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    """Junta párrafos completos hasta llegar a `size` tokens; no corta frases a la mitad."""
    chunks: list[Chunk] = []
    for p in pages:
        paragraphs = [x.strip() for x in p.text.split("\n\n") if x.strip()]
        if len(paragraphs) <= 1:  # PDFs sin líneas en blanco: usa saltos simples
            paragraphs = [x.strip() for x in p.text.split("\n") if x.strip()]
        current: list[str] = []
        i = 0
        for para in paragraphs:
            if current and n_tokens("\n".join(current + [para])) > size:
                text = "\n".join(current)
                chunks.append(Chunk(f"{p.doc_id}-p{p.page}-c{i}", p.source, p.page, text, n_tokens(text)))
                i += 1
                # solapamiento: arrastra los últimos párrafos hasta ~overlap tokens
                tail: list[str] = []
                while current and n_tokens("\n".join(tail)) < overlap:
                    tail.insert(0, current.pop())
                current = tail
            current.append(para)
        if current:
            text = "\n".join(current)
            chunks.append(Chunk(f"{p.doc_id}-p{p.page}-c{i}", p.source, p.page, text, n_tokens(text)))
    return chunks


def save(chunks: list[Chunk], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    pages = load("my_docs")
    # fichas de ~170 tokens: 400 = ficha completa; 128 = cortada a ciegas; párrafos 120 = por sección
    configs = [("fixed", 400, 50), ("fixed", 128, 0), ("paragraph", 120, 0)]
    for name, size, overlap in configs:
        fn = chunk_fixed if name == "fixed" else chunk_by_paragraph
        chunks = fn(pages, size, overlap)
        avg = sum(c.tokens for c in chunks) / max(len(chunks), 1)
        save(chunks, f"data/chunks_{name}_{size}_{overlap}.jsonl")
        print(f"{name:9} size={size:4} overlap={overlap:3} -> {len(chunks):5} chunks, {avg:6.1f} tokens promedio")