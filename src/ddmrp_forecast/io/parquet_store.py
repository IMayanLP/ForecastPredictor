"""Leitura da base Parquet."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
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

def _filtro_data(data_inicio: pd.Timestamp | None):
    return None if data_inicio is None else pc.field("Data") >= pa.scalar(pd.Timestamp(data_inicio), pa.timestamp("ms"))


def contar_vendas_por_serie(arquivos: list[Path], chave: tuple[str, ...],
                            data_inicio: pd.Timestamp | None = None) -> pd.DataFrame:
    # 1ª passada: só chave + Consumo, agregado arquivo a arquivo (pouca memória)
    partes = []
    for arquivo in arquivos:
        tabela = pq.read_table(arquivo, columns=[*chave, "Consumo"], filters=_filtro_data(data_inicio))
        tabela = tabela.append_column("Venda", pc.cast(pc.greater(tabela["Consumo"], 0), pa.int64()))
        partes.append(tabela.group_by(list(chave)).aggregate([("Venda", "sum"), ("Consumo", "count")]).to_pandas())
    return (
        pd.concat(partes, ignore_index=True)
        .groupby(list(chave), sort=False).sum()
        .rename(columns={"Venda_sum": "Vendas", "Consumo_count": "Linhas"})
        .reset_index()
    )


def ler_series_com_venda(pasta: str | Path, clientes: tuple[str, ...] | str | None, chave: tuple[str, ...],
                         data_inicio: pd.Timestamp | None = None, vendas_minimas: int = 1,
                         prefixo_mestre: str = "#", tamanho_lote: int = 2_000_000) -> tuple[pd.DataFrame, dict[str, int]]:
    # séries sem venda desde data_inicio não têm o que prever: ficam fora antes de carregar a base
    # buffers mestres entram sempre (remover_componentes precisa deles para tirar o componente)
    arquivos = filtrar_por_cliente(sorted(Path(pasta).rglob("*.parquet")), clientes)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum Parquet em {pasta} para os clientes {clientes}")

    vendas = contar_vendas_por_serie(arquivos, chave, data_inicio)
    mantidas = vendas.loc[(vendas["Vendas"] >= vendas_minimas)
                          | vendas["Referencia"].str.startswith(prefixo_mestre, na=False), list(chave)]
    chaves = pa.Table.from_pandas(mantidas, preserve_index=False)

    # 2ª passada: lotes de cada arquivo, só linhas das séries mantidas
    lotes = []
    filtro = _filtro_data(data_inicio)
    for arquivo in arquivos:
        leitor = pq.ParquetFile(arquivo)
        # join exige o mesmo tipo nas chaves (pandas gera large_string, o Parquet costuma ter string)
        esquema = leitor.schema_arrow
        chaves_arquivo = chaves.cast(pa.schema([esquema.field(c) for c in chave]))
        for lote in leitor.iter_batches(batch_size=tamanho_lote):
            tabela = pa.Table.from_batches([lote])
            if filtro is not None:
                tabela = tabela.filter(filtro)
            lotes.append(tabela.join(chaves_arquivo, list(chave), join_type="left semi"))
    tabela = pa.concat_tables(lotes)
    for coluna in COLUNAS_CATEGORIA:
        if coluna in tabela.column_names:
            indice = tabela.column_names.index(coluna)
            tabela = tabela.set_column(indice, coluna, pc.dictionary_encode(tabela[coluna]))

    resumo = {
        "séries no período": len(vendas),
        "séries mantidas": len(mantidas),
        f"séries descartadas (< {vendas_minimas} venda(s))": len(vendas) - len(mantidas),
        "linhas no período": int(vendas["Linhas"].sum()),
        "linhas mantidas": tabela.num_rows,
    }
    return tabela.to_pandas(), resumo
