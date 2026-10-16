"""Etapa 6 — features de treino: lags e médias móveis do consumo limpo."""

import pandas as pd

from ddmrp_forecast.features.janelas import agrupar_series, media_movel_defasada


def criar_lags(df: pd.DataFrame, chave: tuple[str, ...], lags: tuple[int, ...],
               coluna: str = "Consumo_Limpo", tipo: str = "float32") -> pd.DataFrame:
    grupo = agrupar_series(df, chave)[coluna]
    for lag in lags:
        df[f"Lag_{lag}"] = grupo.shift(lag).astype(tipo)
    return df


def criar_medias_moveis(df: pd.DataFrame, chave: tuple[str, ...], janelas: tuple[int, ...],
                        coluna: str = "Consumo_Limpo", tipo: str = "float32") -> pd.DataFrame:
    """Media_N = média dos N dias anteriores; NaN até a série ter N dias de histórico."""
    for janela in janelas:
        df[f"Media_{janela}"] = media_movel_defasada(df, chave, coluna, janela).astype(tipo)
    return df
