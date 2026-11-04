"""Descarga los documentos públicos del RAG a my_docs/.

Uso (desde 01-rag/):  uv run python download_docs.py
Los archivos que ya existan no se vuelven a descargar.
"""
import time
import urllib.request
from pathlib import Path

DOCS = {
    # --- Facturación electrónica (DIAN) ---
    "resolucion_dian_0165_2023.pdf": "https://normograma.mintic.gov.co/docs/pdf/resolucion_dian_0165_2023.pdf",
    "anexo_tecnico_factura_electronica_v1_9.pdf": "https://cdn.actualicese.com/normatividad/2023/Resoluciones/Anexo-1-R165-23.pdf",
    "resolucion_dian_0008_2024.pdf": "https://normograma.mintic.gov.co/docs/pdf/resolucion_dian_0008_2024.pdf",
    "resolucion_dian_0012_2021.pdf": "https://normograma.mintic.gov.co/mintic/docs/pdf/resolucion_dian_0012_2021.pdf",
    "resolucion_dian_0037_2021.pdf": "https://normograma.icfes.gov.co/docs/pdf/resolucion_dian_0037_2021.pdf",
    # --- RADIAN: factura electrónica como título valor ---
    "anexo_tecnico_radian_v1_0.pdf": "https://cdn.actualicese.com/normatividad/2021/Resoluciones/Anexo-R000015-21.pdf",
    "concepto_dian_000361_2022_radian.pdf": "https://legalytributarioenlinea.co.pwc.com/Repositorio%20PwC/PDF/Doctrina/DIAN/Conceptos/2022/CON-000361-22.PDF",
    "concepto_dian_000905_2021_radian.pdf": "https://legalytributarioenlinea.co.pwc.com/Repositorio%20PwC/PDF/Doctrina/DIAN/Conceptos/2021/CON-000905-21.PDF",
    "concepto_dian_000625_2021_radian.pdf": "https://legalytributarioenlinea.co.pwc.com/Repositorio%20PwC/PDF/Doctrina/DIAN/Conceptos/CON-000625-21.pdf",
    # --- Documento soporte en adquisiciones a no obligados a facturar ---
    "resolucion_dian_0167_2021.pdf": "https://normograma.icfes.gov.co/docs/pdf/resolucion_dian_0167_2021.pdf",
    "resolucion_dian_0488_2022.pdf": "https://normograma.invima.gov.co/normograma/docs/pdf/resolucion_dian_0488_2022.pdf",
    "anexo_tecnico_documento_soporte_v1_0.pdf": "https://cdn.actualicese.com/normatividad/2020/Resoluciones/Anexo-tecnico-1.0-para-documeto-soporte-compras-no-obligados-facturar.pdf",
    # --- Nómina electrónica ---
    "resolucion_dian_0013_2021_nomina.pdf": "https://legalytributarioenlinea.co.pwc.com/Repositorio%20PwC/PDF/Doctrina/DIAN/Resoluciones/2021/RES-000013-21.pdf",
    "concepto_dian_001268_2021_nomina.pdf": "https://legalytributarioenlinea.co.pwc.com/Repositorio%20PwC/PDF/Doctrina/DIAN/Conceptos/2021/CON-001268-21.PDF",
    "concepto_dian_000589_2021_nomina.pdf": "https://legalytributarioenlinea.co.pwc.com/Repositorio%20PwC/PDF/Doctrina/DIAN/Conceptos/2022/CON-000589-21.PDF",
    "concepto_dian_000878_2022_nomina.pdf": "https://legalytributarioenlinea.co.pwc.com/Repositorio%20PwC/PDF/Doctrina/DIAN/Conceptos/2022/CON-000878-22.pdf",
    "concepto_dian_0311_2021_nomina.pdf": "https://cdn.actualicese.com/normatividad/2021/Conceptos/C311-21.pdf",
    # --- Laboral ---
    "ley_2466_2025_reforma_laboral.pdf": "https://www.funcionpublica.gov.co/eva/gestornormativo/norma_pdf.php?i=260676",
    "ley_2101_2021_jornada_laboral.pdf": "https://www.funcionpublica.gov.co/eva/gestornormativo/norma_pdf.php?i=166506",
    "concepto_funcion_publica_jornada_a.pdf": "https://www.funcionpublica.gov.co/eva/gestornormativo/norma_pdf.php?i=274636",
    "concepto_funcion_publica_jornada_b.pdf": "https://www.funcionpublica.gov.co/eva/gestornormativo/norma_pdf.php?i=267200",
    # --- Protección de datos personales ---
    "ley_1581_2012_datos_personales.pdf": "https://www.funcionpublica.gov.co/eva/gestornormativo/norma_pdf.php?i=49981",
    "decreto_1377_2013_datos_personales.pdf": "https://www.funcionpublica.gov.co/eva/gestornormativo/norma_pdf.php?i=53646",
}

OUT = Path("my_docs")
HEADERS = {"User-Agent": "Mozilla/5.0 (descarga de documentos para estudio)"}


def download(name: str, url: str) -> str:
    target = OUT / name
    if target.exists() and target.stat().st_size > 0:
        return "ya existía"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):  # algunas páginas devuelven HTML en vez del PDF
        raise ValueError("la respuesta no es un PDF")
    target.write_bytes(data)
    return f"{len(data) / 1e6:.1f} MB"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    failed = []
    for name, url in DOCS.items():
        try:
            print(f"OK    {name}: {download(name, url)}")
        except Exception as e:  # sigue con los demás si uno falla
            failed.append((name, url))
            print(f"FALLO {name}: {e}")
        time.sleep(1)  # sin prisa: son servidores públicos
    print(f"\n{len(DOCS) - len(failed)}/{len(DOCS)} documentos en {OUT}/")
    for name, url in failed:
        print(f"  Descárgalo a mano y guárdalo como my_docs/{name}: {url}")