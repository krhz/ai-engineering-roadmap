"""Convierte cada campaña del repo data-pa-la-democracia en una ficha de texto para el RAG.

run uv run python -m rag.fichas ../data-pa-la-democracia
"""
import re
import sys
from pathlib import Path

import pandas as pd

CARGOS = {"Alcaldia_fun": "Alcaldía", "Gobernación": "Gobernación"}
AGRUPACION = {"Agrupacion Politica": "partido o movimiento", "Coalicion Politica": "coalición",
              "Grupo Politico": "grupo significativo de ciudadanos"}


def pesos(x) -> str:
    return "no reportado" if pd.isna(x) else "$" + f"{x:,.0f}".replace(",", ".")


def nombre(snake: str) -> str:
    return " ".join(w.capitalize() for w in snake.split("_"))


def ficha(r: pd.Series, gastos: list[str], ingresos: list[str]) -> str:
    cargo = CARGOS[r["Cargo"]]
    lugar = (f"la Alcaldía de {r['Ciudades']} ({r['Departamento']})" if cargo == "Alcaldía"
             else f"la Gobernación de {r['Departamento']}")
    lines = [f"Campaña de {nombre(r['Nombre'])} a {lugar}, elecciones territoriales de 2023.",
             f"Agrupación: {r['Nombre de Agrupación']} ({AGRUPACION.get(r['Tipo de Agrupación'], r['Tipo de Agrupación'])}).",
             "",
             f"Ingresos reportados: {pesos(r['Total de Ingresos'])} en total."]
    lines += [f"- {c}: {pesos(r[c])}" for c in ingresos if r[c] > 0]
    lines += ["", f"Gastos reportados: {pesos(r['Total de Gastos'])} en total."]
    lines += [f"- {c}: {pesos(r[c])}" for c in gastos if r[c] > 0]
    if cargo == "Alcaldía" and pd.notna(r["TOPE GASTOS"]):
        pct = f"{100 * r['Total de Gastos'] / r['TOPE GASTOS']:.1f}".replace(".", ",")
        lines += ["", f"Tope de gastos para la alcaldía de {r['Ciudades']}: {pesos(r['TOPE GASTOS'])}. "
                      f"El gasto reportado equivale al {pct} % del tope."]
    return "\n".join(lines)


if __name__ == "__main__":
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else "../data-pa-la-democracia")
    df = pd.read_csv(repo / "campañas_anomalias_resultados.csv")
    df = df.drop_duplicates().reset_index(drop=True)  # el merge del repo duplica algunas filas
    gastos = [c for c in df.columns if re.match(r"^2\d\d - ", c)]
    ingresos = [c for c in df.columns if re.match(r"^1\d\d - ", c)]
    df[gastos + ingresos] = df[gastos + ingresos].fillna(0)

    out = pd.DataFrame({
        "fila": range(1, len(df) + 1),  # hace las veces de "página" en las citas
        "texto": [ficha(r, gastos, ingresos) for _, r in df.iterrows()],
        "nombre": df["Nombre"].map(nombre),
        "cargo": df["Cargo"].map(CARGOS),
        "departamento": df["Departamento"],
        "municipio": df["Ciudades"].where(df["Cargo"] == "Alcaldia_fun"),
        "agrupacion": df["Nombre de Agrupación"],
        "total_ingresos": df["Total de Ingresos"],
        "total_gastos": df["Total de Gastos"],
        "tope_gastos": df["TOPE GASTOS"],
    })
    for c in gastos + ingresos:  # columnas g201..g211 e i101..i107, para SQL en la semana 2
        out[("g" if c[0] == "2" else "i") + c[:3]] = df[c]
    Path("my_docs").mkdir(exist_ok=True)
    out.to_csv("my_docs/campanas_2023.csv", index=False)
    print(f"{len(out)} fichas en my_docs/campanas_2023.csv")
    print(out.texto.iloc[0])