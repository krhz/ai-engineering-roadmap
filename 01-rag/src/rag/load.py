from dataclasses import dataclass
from pathlib import Path
import unicodedata

import docx  # python-docx
import pandas as pd
import pymupdf  # PyMuPDF


@dataclass
class Page:
    doc_id: str   # nombre del archivo sin extensión
    source: str   # nombre del archivo, para la cita
    page: int     # número de página (1 en Word y Markdown)
    text: str


def load(folder: str) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(Path(folder).rglob("*")):
        ext = path.suffix.lower()
        if ext == ".csv":  # fichas: una "página" por fila, con columnas fila y texto
            df = pd.read_csv(path, dtype=str)
            for row in df.itertuples():
                pages.append(Page(path.stem, path.name, int(row.fila), row.texto))
        elif ext == ".pdf":
            with pymupdf.open(path) as pdf:
                for number, pdf_page in enumerate(pdf, start=1):
                    # NFKC convierte espacios no separables (\xa0) y similares en caracteres normales
                    text = unicodedata.normalize("NFKC", pdf_page.get_text()).strip()
                    if text:  # páginas vacías o escaneadas no traen texto
                        pages.append(Page(path.stem, path.name, number, text))
        elif ext == ".docx":
            document = docx.Document(str(path))
            text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
            pages.append(Page(path.stem, path.name, 1, text))
        elif ext in (".md", ".txt"):
            pages.append(Page(path.stem, path.name, 1, path.read_text(encoding="utf-8")))
    return pages


if __name__ == "__main__":
    pages = load("my_docs")
    docs = {p.source for p in pages}
    print(f"{len(docs)} documentos, {len(pages)} páginas con texto")
    for p in pages[:3]:
        print(f"--- {p.source} p.{p.page} ({len(p.text)} caracteres)")
        print(p.text[:300])