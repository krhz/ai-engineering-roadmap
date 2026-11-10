"""Arma eval/questions.csv (35 preguntas) y eval/questions_agregadas.csv (5, para la semana 2) desde las fichas.

Las respuestas esperadas salen de los datos con pandas, no se escriben a mano.
Uso: uv run python -m rag.questions
"""
import pandas as pd

from rag.fichas import pesos

# cómo diría un usuario cada código de gasto
COLOQUIAL = {"g208": "publicidad y propaganda", "g205": "transporte", "g204": "eventos y actos públicos",
             "g201": "administración de la campaña", "g203": "materiales y publicaciones"}


def num(x) -> str:
    return pesos(x).lstrip("$")  # "18.200.000": así aparece en la ficha


def donde(r) -> str:
    return f"la alcaldía de {r.municipio}" if r.cargo == "Alcaldía" else f"la gobernación de {r.departamento}"


df = pd.read_csv("my_docs/campanas_2023.csv")
unicos = df[~df.nombre.duplicated(keep=False) & df.total_gastos.gt(0) & df.total_ingresos.gt(0)]
rows = []

# literal: el total de una campaña (5 de gastos y 5 de ingresos; 2 de gobernación)
lit = pd.concat([unicos[unicos.cargo == "Alcaldía"].sample(8, random_state=1),
                 unicos[unicos.cargo == "Gobernación"].sample(2, random_state=1)])
for i, r in enumerate(lit.itertuples()):
    if i % 2 == 0:
        rows.append((f"¿Cuánto gastó en total la campaña de {r.nombre} a {donde(r)}?", num(r.total_gastos), r.fila, "literal"))
    else:
        rows.append((f"¿Cuánta plata recibió {r.nombre} para su campaña a {donde(r)}?", num(r.total_ingresos), r.fila, "literal"))

# codigo: un rubro de gasto puntual
usados = set(lit.fila)
cod = unicos[~unicos.fila.isin(usados)].sample(frac=1, random_state=2)
codigos = list(COLOQUIAL)
for r in cod.itertuples():
    c = codigos[len([x for x in rows if x[3] == "codigo"]) % len(codigos)]
    if getattr(r, c) > 0:
        rows.append((f"¿Cuánto invirtió {r.nombre} en {COLOQUIAL[c]} en su campaña a {donde(r)}?", num(getattr(r, c)), r.fila, "codigo"))
        usados.add(r.fila)
    if len([x for x in rows if x[3] == "codigo"]) == 10:
        break

# combinada: comparar dos campañas del mismo municipio (hay que recuperar las dos fichas)
alc = unicos[(unicos.cargo == "Alcaldía") & ~unicos.fila.isin(usados)]
pares = alc.groupby(["departamento", "municipio"]).filter(lambda g: len(g) >= 2)
grupos = list(pares.groupby(["departamento", "municipio"]))
for k in pd.Series(range(len(grupos))).sample(frac=1, random_state=3):  # municipios al azar, no en orden alfabético
    (dep, mun), g = grupos[k]
    a, b = g.sample(2, random_state=3).itertuples()
    if a.total_gastos == b.total_gastos:
        continue
    win = a if a.total_gastos > b.total_gastos else b
    rows.append((f"En la elección de alcaldía de {mun} ({dep}), ¿quién gastó más en campaña: {a.nombre} o {b.nombre}?",
                 win.nombre, win.fila, "combinada"))
    if len([x for x in rows if x[3] == "combinada"]) == 10:
        break

qs = pd.DataFrame(rows, columns=["question", "expected", "page", "type"])
qs = pd.concat([qs[qs.type == t] for t in ("literal", "combinada", "codigo")]).reset_index(drop=True)
qs["source"] = "campanas_2023.csv"
sin = pd.DataFrame({"question": [
    "¿Cuánto gastó la campaña al Concejo de Barranquilla del Partido Liberal en 2023?",
    "¿Cuánto costó la campaña al Senado de 2022 del Pacto Histórico?",
    "¿Cuánto gastaron los candidatos a la alcaldía de Medellín en las elecciones de 2019?",
    "¿Quién ganó la alcaldía de Cartagena en 2023?",
    "¿Cuánto gastaron las campañas a la Asamblea de Antioquia en 2023?"],
    "expected": "NO_SE", "source": "", "page": "", "type": "sin_respuesta"})
qs = pd.concat([qs, sin]).reset_index(drop=True)
qs.insert(0, "id", [f"q{i:02d}" for i in range(1, len(qs) + 1)])
qs[["id", "question", "expected", "source", "page", "type"]].to_csv("eval/questions.csv", index=False)

# agregadas: necesitan sumar o contar muchas filas; el RAG de la semana 1 no puede, SQL sí (semana 2)
a = df[df.cargo == "Alcaldía"]
top_bq = a[a.municipio == "Barranquilla"].sort_values("total_gastos").iloc[-1]
agg = pd.DataFrame([
    ("¿Cuántas campañas a alcaldía hubo en Atlántico?", str((a.departamento == "Atlántico").sum())),
    ("¿Cuánto gastaron en total todas las campañas a gobernación?", num(df[df.cargo == "Gobernación"].total_gastos.sum())),
    ("¿Qué campaña a la alcaldía de Barranquilla reportó más gastos?", top_bq.nombre),
    ("¿Cuántas campañas a alcaldía reportaron gastos por encima del tope?", str((a.total_gastos > a.tope_gastos).sum())),
    ("¿Cuál fue el gasto promedio de las campañas a alcaldía en Antioquia?", num(a[a.departamento == "Antioquia"].total_gastos.mean())),
], columns=["question", "expected"])
agg.insert(0, "id", [f"a{i:02d}" for i in range(1, 6)])
agg["type"] = "agregada"
agg.to_csv("eval/questions_agregadas.csv", index=False)
print(qs.type.value_counts().to_string())
print(f"{len(agg)} preguntas agregadas en eval/questions_agregadas.csv")