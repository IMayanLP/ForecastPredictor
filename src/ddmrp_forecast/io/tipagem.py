"""Tipagem comum das colunas exportadas do SSMS (.rpt e .csv)."""

import codecs
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

VALORES_NULOS = pa.array(["", "NULL"])

COLUNAS_DATA = ["Data"]
COLUNAS_INTEIRAS = ["DiaSemana", "DiaMes", "SemanaAno", "Mes", "TrimestreAno", "Ano"]
COLUNAS_DECIMAIS = [
    "Stock", "StockEnTransito", "StockTotal", "Consumo",
    "ADU", "DiasADU", "ADI", "CV", "TiempoReposicion",
]


def detectar_encoding(caminho: Path) -> str:
    with open(caminho, "rb") as f:
        inicio = f.read(4)
    if inicio.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if inicio.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"  # padrão do SSMS
    return "utf-8"


def tipar_coluna(nome: str, valores: pa.Array) -> pa.Array:
    valores = pc.utf8_trim_whitespace(valores)
    valores = pc.if_else(pc.is_in(valores, value_set=VALORES_NULOS), pa.scalar(None, pa.string()), valores)

    if nome in COLUNAS_DATA:
        # "2026-01-01 00:00:00.000" → só a data
        datas = pc.cast(pc.utf8_slice_codeunits(valores, 0, 10), pa.date32())
        return pc.cast(datas, pa.timestamp("s"))
    if nome in COLUNAS_INTEIRAS:
        return pc.cast(valores, pa.int16())
    if nome in COLUNAS_DECIMAIS:
        return pc.cast(valores, pa.float64())
    return valores
