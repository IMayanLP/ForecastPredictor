"""Leitura de .csv exportado do SSMS (com cabeçalho), em lotes."""

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

from ddmrp_forecast.io.tipagem import COLUNAS_DECIMAIS, detectar_encoding, tipar_coluna

BYTES_POR_LINHA = 200  # estimativa para converter tamanho_lote em bytes de leitura


def ler_csv_em_lotes(origem: Path, tamanho_lote: int = 100_000) -> Iterator[pa.Table]:
    encoding = detectar_encoding(origem)
    with open(origem, encoding=encoding) as f:
        cabecalho = f.readline().rstrip("\r\n")

    # ";" = CSV regional (Excel pt-BR/es), com vírgula decimal
    separador = ";" if cabecalho.count(";") > cabecalho.count(",") else ","
    colunas = [c.strip().strip('"').strip() for c in cabecalho.split(separador)]
    if "Data" not in colunas:
        raise ValueError(f"{origem.name}: cabeçalho sem a coluna Data — exportar o CSV com cabeçalho")

    leitor = pv.open_csv(
        origem,
        read_options=pv.ReadOptions(
            encoding="utf8" if encoding.startswith("utf-8") else encoding,
            column_names=colunas,
            skip_rows=1,
            block_size=max(tamanho_lote * BYTES_POR_LINHA, 1 << 20),
        ),
        parse_options=pv.ParseOptions(delimiter=separador),
        convert_options=pv.ConvertOptions(column_types={c: pa.string() for c in colunas}),
    )
    for lote in leitor:
        arrays = []
        for nome, valores in zip(colunas, lote.columns):
            if separador == ";" and nome in COLUNAS_DECIMAIS:
                valores = pc.replace_substring(valores, ",", ".")
            arrays.append(tipar_coluna(nome, valores))
        yield pa.table(arrays, names=colunas)
