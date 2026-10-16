"""Etapa 2 — preparação inicial (porte do bloco do Colab)."""

import numpy as np
import pandas as pd

from ddmrp_forecast.config.settings import CHAVE_SERIE
from ddmrp_forecast.features.janelas import agrupar_series
from ddmrp_forecast.preprocess.buffers_mestre import remover_componentes


def preparar_base(df: pd.DataFrame, escopo_buffer_mestre: tuple[str, ...] = ("Cliente",)) -> tuple[pd.DataFrame, int]:
    # o strip dos textos (Unidad etc.) já é feito na conversão para Parquet
    df, linhas_removidas = remover_componentes(df, escopo_buffer_mestre)
    # category ordena pela ordem de aparição no Parquet; categorias em ordem alfabética
    for coluna in CHAVE_SERIE:
        if isinstance(df[coluna].dtype, pd.CategoricalDtype):
            df[coluna] = df[coluna].cat.reorder_categories(sorted(df[coluna].cat.categories))
    df = df.sort_values([*CHAVE_SERIE, "Data"]).reset_index(drop=True)
    return df, linhas_removidas


def verificar_ordenacao(df: pd.DataFrame, chave: tuple[str, ...] = CHAVE_SERIE) -> None:
    """Lags e médias móveis dependem disto: cada série em linhas contíguas e Data crescente dentro dela."""
    if len(df) < 2:
        return
    codigos = agrupar_series(df, chave).ngroup().to_numpy()
    mesma_serie = codigos[1:] == codigos[:-1]

    blocos = int((~mesma_serie).sum()) + 1
    series = int(codigos.max()) + 1
    if blocos != series:
        raise ValueError(f"Séries fora de ordem: {series:,} séries espalhadas em {blocos:,} blocos de linhas")

    inicio_series = np.flatnonzero(np.r_[True, ~mesma_serie])
    chaves = df[list(chave)].iloc[inicio_series].astype(str).reset_index(drop=True)
    if not chaves.equals(chaves.sort_values(list(chave)).reset_index(drop=True)):
        raise ValueError(f"Séries fora da ordem alfabética de {list(chave)}")

    datas = df["Data"].to_numpy()
    fora_de_ordem = int(np.sum(datas[1:][mesma_serie] <= datas[:-1][mesma_serie]))
    if fora_de_ordem:
        raise ValueError(f"{fora_de_ordem:,} linhas com Data fora de ordem (ou repetida) dentro da série")
