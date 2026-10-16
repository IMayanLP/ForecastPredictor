"""Operações em janela dentro de cada série (df ordenado por série e Data)."""

import pandas as pd


def agrupar_series(df: pd.DataFrame, chave: tuple[str, ...]):
    return df.groupby(list(chave), observed=True, sort=False)


def media_movel_defasada(df: pd.DataFrame, chave: tuple[str, ...], coluna: str,
                         janela: int, min_periodos: int | None = None) -> pd.Series:
    """Média dos `janela` registros anteriores ao atual (shift 1), sem cruzar séries.

    Equivale a groupby(chave)[coluna].transform(lambda x: x.shift(1).rolling(...).mean()),
    sem o lambda por grupo.
    """
    grupos = agrupar_series(df, chave)
    anterior = grupos[coluna].shift(1)
    return media_movel_por_codigo(anterior, grupos.ngroup(), janela,
                                  janela if min_periodos is None else min_periodos)


def media_movel_por_codigo(valores: pd.Series, codigos: pd.Series, janela: int,
                           min_periodos: int) -> pd.Series:
    media = (
        valores
        .groupby(codigos, sort=False)
        .rolling(janela, min_periods=min_periodos)
        .mean()
    )
    return media.droplevel(0)
