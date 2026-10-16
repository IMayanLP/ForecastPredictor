"""Etapa 4 — calendário diário completo por série (dias sem registro = consumo 0)."""

import numpy as np
import pandas as pd

from ddmrp_forecast.features.janelas import agrupar_series

UM_DIA = np.timedelta64(1, "D")


def completar_calendario(df: pd.DataFrame, chave: tuple[str, ...],
                         colunas_fixas: list[str]) -> tuple[pd.DataFrame, int]:
    """Uma linha por série e dia, do primeiro ao último registro da série.

    Retorna (df completo, dias adicionados). Exige no máximo um registro por série e dia.
    """
    chave = list(chave)
    duplicados = int(df.duplicated([*chave, "Data"]).sum())
    if duplicados:
        raise ValueError(f"{duplicados:,} registros duplicados por {[*chave, 'Data']}")

    grupos = agrupar_series(df, chave)
    codigos = grupos.ngroup().to_numpy()
    inicio = grupos["Data"].min().to_numpy()
    fim = grupos["Data"].max().to_numpy()

    dias_por_serie = (fim - inicio) // UM_DIA + 1
    deslocamento = np.concatenate([[0], np.cumsum(dias_por_serie)[:-1]])
    total = int(dias_por_serie.sum())

    # posição de cada registro existente na tabela completa (sem merge)
    posicao = deslocamento[codigos] + (df["Data"].to_numpy() - inicio[codigos]) // UM_DIA
    completo = df.set_index(pd.Index(posicao)).reindex(pd.RangeIndex(total))

    # chave e Data em todas as linhas, inclusive nas novas
    serie_da_linha = np.repeat(np.arange(len(dias_por_serie)), dias_por_serie)
    dia_na_serie = np.arange(total) - deslocamento[serie_da_linha]
    completo["Data"] = (inicio[serie_da_linha] + dia_na_serie * UM_DIA).astype(df["Data"].dtype)

    primeira_linha = np.unique(codigos, return_index=True)[1]
    for coluna in chave:
        valores = df[coluna].iloc[primeira_linha].reset_index(drop=True)
        completo[coluna] = valores.take(serie_da_linha).set_axis(completo.index)

    completo["Consumo"] = completo["Consumo"].fillna(0)
    completo["EhPico"] = completo["EhPico"].astype("boolean").fillna(False).astype(bool)
    completo["Consumo_Limpo"] = completo["Consumo_Limpo"].fillna(0)

    fixas = [c for c in colunas_fixas if c in completo.columns]
    completo[fixas] = completo[fixas].groupby(serie_da_linha).ffill()
    completo[fixas] = completo[fixas].groupby(serie_da_linha).bfill()

    return completo, total - len(df)
