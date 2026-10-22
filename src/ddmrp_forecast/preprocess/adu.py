"""Etapa 5 — ADU recalculado: média do Consumo nos DiasADU dias anteriores."""

import numpy as np
import pandas as pd

from ddmrp_forecast.features.janelas import agrupar_series, media_movel_por_codigo


def corrigir_dias_adu(df: pd.DataFrame, padrao: float = 30.0) -> tuple[pd.DataFrame, int]:
    zerados = df["DiasADU"] == 0
    df["DiasADU"] = df["DiasADU"].mask(zerados, padrao)
    return df, int(zerados.sum())


def recalcular_adu(df: pd.DataFrame, chave: tuple[str, ...], coluna: str = "Consumo",
                   nome: str = "ADU_recalculado") -> pd.DataFrame:
    grupos = agrupar_series(df, chave)
    dias_adu = np.trunc(grupos["DiasADU"].transform("first"))
    anterior = grupos[coluna].shift(1)
    codigos = grupos.ngroup()

    adu = pd.Series(np.nan, index=df.index)
    for dias in dias_adu[dias_adu > 0].unique():
        linhas = dias_adu == dias
        adu[linhas] = media_movel_por_codigo(anterior[linhas], codigos[linhas], int(dias), int(dias))
    df[nome] = adu
    return df
