"""Etapa 2 — preparação inicial (porte do bloco do Colab)."""

import pandas as pd

from ddmrp_forecast.preprocess.buffers_mestre import remover_componentes

CHAVE_SERIE = ["Cliente", "IdBodega", "Referencia"]


def preparar_base(df: pd.DataFrame, escopo_buffer_mestre: tuple[str, ...] = ("Cliente",)) -> tuple[pd.DataFrame, int]:
    # o strip dos textos (Unidad etc.) já é feito na conversão para Parquet
    df, linhas_removidas = remover_componentes(df, escopo_buffer_mestre)
    df = df.sort_values([*CHAVE_SERIE, "Data"]).reset_index(drop=True)
    return df, linhas_removidas