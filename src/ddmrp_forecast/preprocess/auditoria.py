"""Auditoria da base final: refaz picos e ADU por outro caminho e compara."""

import numpy as np
import pandas as pd

from ddmrp_forecast.features.janelas import agrupar_series

TOLERANCIA = 1e-9


def _diferentes(obtido: np.ndarray, esperado: np.ndarray) -> int:
    nulo_o, nulo_e = np.isnan(obtido), np.isnan(esperado)
    valores = ~nulo_o & ~nulo_e & ~np.isclose(obtido, esperado, rtol=TOLERANCIA, atol=TOLERANCIA)
    return int((nulo_o != nulo_e).sum() + valores.sum())


def auditar_picos(df: pd.DataFrame, janela_limpeza: int = 7) -> tuple[dict[str, int], dict[str, int]]:
    media = df[f"Media_{janela_limpeza}_Limpa"].to_numpy(float)
    esperado = np.where(df["EhPico"], np.nan_to_num(media, nan=0.0), df["Consumo"].to_numpy(float))
    limpo = df["Consumo_Limpo"].to_numpy(float)
    limite = df["LimiteSuperior"].to_numpy(float)
    informacoes = {
        "picos": int(df["EhPico"].sum()),
        "picos sem histórico (viram 0)": int((df["EhPico"] & np.isnan(media)).sum()),
    }
    erros = {
        "Consumo_Limpo diferente do esperado": _diferentes(limpo, esperado),
        "Consumo_Limpo acima do LimiteSuperior": int(np.nansum(limpo > limite)),
    }
    return informacoes, erros


def auditar_adu(df: pd.DataFrame, chave: tuple[str, ...]) -> tuple[dict[str, int], dict[str, int]]:
    # média dos d dias anteriores = (soma acumulada até t-1 − soma acumulada até t-d-1) / d
    grupos = agrupar_series(df, chave)
    dias = np.trunc(grupos["DiasADU"].transform("first").to_numpy(float))
    posicao = grupos.cumcount().to_numpy()
    acumulado = grupos["Consumo"].cumsum().to_numpy(float)

    i = np.arange(len(df))
    d = np.nan_to_num(dias, nan=0).astype(np.int64)
    tem_janela = (d > 0) & (posicao >= d)
    ate_ontem = np.where(posicao >= 1, acumulado[np.maximum(i - 1, 0)], 0.0)
    antes_da_janela = np.where(posicao - d - 1 >= 0, acumulado[np.maximum(i - d - 1, 0)], 0.0)
    esperado = np.where(tem_janela, (ate_ontem - antes_da_janela) / np.maximum(d, 1), np.nan)

    series_sem_dias = grupos["DiasADU"].count().eq(0)
    informacoes = {
        "linhas com ADU recalculado": int(df["ADU_recalculado"].notna().sum()),
        "linhas sem histórico suficiente (< DiasADU dias)": int(((d > 0) & (posicao < d)).sum()),
        "séries sem DiasADU": int(series_sem_dias.sum()),
    }
    erros = {
        "linhas com DiasADU = 0": int((df["DiasADU"] == 0).sum()),
        "ADU_recalculado diferente do esperado": _diferentes(df["ADU_recalculado"].to_numpy(float), esperado),
    }
    return informacoes, erros
