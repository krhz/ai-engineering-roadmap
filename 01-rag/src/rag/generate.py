import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from rag.embed import LocalEmbedder
from rag.index import Reranker, connect, search

load_dotenv()

SYSTEM = """Respondes preguntas sobre los ingresos y gastos que las campañas a alcaldías y gobernaciones de 2023
reportaron al CNE, usando SOLO las fichas numeradas que te doy.
Reglas:
- Si las fichas no contienen la respuesta, pon found=false y answer="No lo sé con los documentos disponibles."
- Si la pregunta pide contar, sumar o promediar muchas campañas (un departamento entero, "todas", "cuántas"),
  pon found=false: con 5 fichas no se puede calcular bien.
- Copia las cifras tal como aparecen en la ficha, con sus puntos de miles.
- Reporta cifras; no califiques a ninguna campaña ni persona de irregular.
- No uses conocimiento propio ni inventes datos.
- En citations pon los números de las fichas que usaste, por ejemplo [1, 3].
- Responde en español, en 1 a 3 frases."""


class Answer(BaseModel):
    found: bool = Field(description="True si los fragmentos contienen la respuesta")
    answer: str
    citations: list[int] = Field(default_factory=list, description="Números de los fragmentos usados")


class InvalidCitation(Exception):
    pass


llm = OpenAI(api_key=os.environ.get("LLM_API_KEY", "x"), base_url=os.environ.get("LLM_BASE_URL"))

def build_context(hits: list[tuple]) -> str:
    # hits = filas (chunk_id, source, page, content)
    return "\n\n".join(f"[{i}] ({h[1]} fila {h[2]})\n{h[3]}" for i, h in enumerate(hits, start=1))


def generate(question: str, hits: list[tuple]) -> Answer:
    completion = llm.chat.completions.parse(
        model=os.environ["CHAT_MODEL"],
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Fragmentos:\n{build_context(hits)}\n\nPregunta: {question}"},
        ],
        response_format=Answer,  # el SDK pide JSON con este esquema y lo valida con Pydantic
        temperature=0,
    )
    answer = completion.choices[0].message.parsed
    bad = [n for n in answer.citations if not 1 <= n <= len(hits)]
    if bad:
        raise InvalidCitation(f"El modelo citó fragmentos que no existen: {bad}")
    if answer.found and not answer.citations:
        raise InvalidCitation("Respondió sin citar ningún fragmento")
    return answer


def render(answer: Answer, hits: list[tuple]) -> str:
    if not answer.found:
        return answer.answer
    sources = "  ".join(f"[{n}] {hits[n - 1][1]} fila {hits[n - 1][2]}" for n in sorted(set(answer.citations)))
    return f"{answer.answer}\n\nFuentes: {sources} (reportes al CNE en Cuentas Claras, exportados el 9 de marzo de 2024)"


_state: dict = {}


def rag(question: str, mode: str = "rerank") -> tuple[Answer, list[tuple]]:
    if not _state:  # conecta y carga modelos una sola vez
        _state.update(conn=connect(), embedder=LocalEmbedder(), reranker=Reranker())
    hits = search(_state["conn"], _state["embedder"], question, mode, k=5, reranker=_state["reranker"])
    return generate(question, hits), hits


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "¿Cuánto gastó en total la campaña de Alejandro Char Chaljub a la alcaldía de Barranquilla?"
    answer, hits = rag(question)
    print(render(answer, hits))