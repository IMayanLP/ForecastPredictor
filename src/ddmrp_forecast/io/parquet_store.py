"""Leitura da base Parquet."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ddmrp_forecast.io.selecao import filtrar_por_cliente

# Textos repetidos lidos como categoria: ocupam bem menos memória
COLUNAS_CATEGORIA = ["Cliente", "IdBodega", "Referencia", "Descripcion", "Unidad"]


def ler_parquet(pasta: str | Path, colunas: list[str] | None = None,
                filtros: list[tuple] | None = None,
                clientes: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Lê os Parquet da pasta (e subpastas) num único DataFrame.

    colunas: lê só as colunas indicadas.
    filtros: ex. [("Cliente", "==", "ACME")] — filtra antes de carregar.
    clientes: lista branca — só arquivos cujo nome contém um dos clientes. None = todos.
    """
    arquivos = filtrar_por_cliente(sorted(Path(pasta).rglob("*.parquet")), clientes)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum Parquet em {pasta} para os clientes {clientes}")

    tabela = pq.read_table(
        [str(a) for a in arquivos],
        columns=colunas,
        filters=filtros,
        read_dictionary=[c for c in COLUNAS_CATEGORIA if colunas is None or c in colunas],
    )
    return tabela.to_pandas()