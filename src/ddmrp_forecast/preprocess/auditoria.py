"""Auditoria da base final: refaz picos e ADU por outro caminho e compara."""

import numpy as np
import pandas as pd

from ddmrp_forecast.features.janelas import agrupar_series

TOLERANCIA = 1e-9


def _diferentes(obtido: np.ndarray, esperado: np.ndarray) -> int:
    nulo_o, nulo_e = np.isnan(obtido), np.isnan(esperado)
    valores = ~nulo_o & ~nulo_e & ~np.isclose(obtido, esperado, rtol=TOLERANCIA, atol=TOLERANCIA)
    return int((nulo_o != nulo_e).sum() + valores.sum())


def auditar_picos(df: pd.DataFrame, janela_limpeza: int = 7,
                  substituicao: str = "media") -> tuple[dict[str, int], dict[str, int]]:
    limpo = df["Consumo_Limpo"].to_numpy(float)
    limite = df["LimiteSuperior"].to_numpy(float)
    informacoes = {"picos": int(df["EhPico"].sum())}
    if substituicao == "limite":
        esperado = np.where(df["EhPico"], limite, df["Consumo"].to_numpy(float))
        volume_bruto = float(df["Consumo"].sum())
        informacoes["volume cortado nos picos (% do Consumo)"] = round(
            (volume_bruto - float(np.nansum(limpo))) / volume_bruto * 100 if volume_bruto else 0.0, 1)
    else:
        media = df[f"Media_{janela_limpeza}_Limpa"].to_numpy(float)
        esperado = np.where(df["EhPico"], np.nan_to_num(media, nan=0.0), df["Consumo"].to_numpy(float))
        informacoes["picos sem histórico (viram 0)"] = int((df["EhPico"] & np.isnan(media)).sum())
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


def auditar_features(df: pd.DataFrame, chave: tuple[str, ...], lags: tuple[int, ...], janelas: tuple[int, ...],
                     coluna: str = "Consumo_Limpo") -> tuple[dict[str, int], dict[str, int]]:
    # refaz Lag_N e Media_N por posição/soma acumulada sobre `coluna` e conta onde o Consumo bruto daria outro valor
    grupos = agrupar_series(df, chave)
    posicao = grupos.cumcount().to_numpy()
    i = np.arange(len(df))
    erros, informacoes = {}, {}
    for nome_coluna, valores in [(coluna, df[coluna].to_numpy(float)), ("Consumo", df["Consumo"].to_numpy(float))]:
        acumulado = grupos[nome_coluna].cumsum().to_numpy(float)
        for lag in lags:
            esperado = np.where(posicao >= lag, valores[np.maximum(i - lag, 0)], np.nan)
            obtido = df[f"Lag_{lag}"].to_numpy(float)
            if nome_coluna == coluna:
                erros[f"Lag_{lag} diferente de {coluna}"] = _diferentes_float32(obtido, esperado)
            else:
                informacoes[f"Lag_{lag} ≠ Consumo bruto (picos excluídos)"] = _diferentes_float32(obtido, esperado)
        for janela in janelas:
            ate_ontem = np.where(posicao >= 1, acumulado[np.maximum(i - 1, 0)], 0.0)
            antes = np.where(posicao - janela - 1 >= 0, acumulado[np.maximum(i - janela - 1, 0)], 0.0)
            esperado = np.where(posicao >= janela, (ate_ontem - antes) / janela, np.nan)
            obtido = df[f"Media_{janela}"].to_numpy(float)
            if nome_coluna == coluna:
                erros[f"Media_{janela} diferente de {coluna}"] = _diferentes_float32(obtido, esperado)
            else:
                informacoes[f"Media_{janela} ≠ Consumo bruto (picos excluídos)"] = _diferentes_float32(obtido, esperado)
    return informacoes, erros


def _diferentes_float32(obtido: np.ndarray, esperado: np.ndarray) -> int:
    # features ficam em float32: tolerância relativa do float32
    nulo_o, nulo_e = np.isnan(obtido), np.isnan(esperado)
    valores = ~nulo_o & ~nulo_e & ~np.isclose(obtido, esperado, rtol=1e-5, atol=1e-5)
    return int((nulo_o != nulo_e).sum() + valores.sum())
