"""Partes da Data com a mesma semântica do DATEPART do SQL Server."""

import pandas as pd

TIPO_PARTES = "int16"


def partes_da_data(datas: pd.Series, datefirst: int = 7) -> pd.DataFrame:
    """DiaSemana, DiaMes, SemanaAno, Mes, TrimestreAno, Ano como o SQL Server calcula.

    datefirst: SET DATEFIRST do servidor (7 = domingo, padrão us_english).
    WEEKDAY: 1 = dia `datefirst`. WEEK: semana 1 contém 1º/jan e cada semana começa em `datefirst`.
    """
    dt = datas.dt
    dia_semana = (dt.dayofweek + 1 - datefirst) % 7 + 1   # dayofweek: segunda = 0
    dia_semana_1jan = (dia_semana - 1 - (dt.dayofyear - 1)) % 7 + 1
    semana = (dt.dayofyear - 1 + dia_semana_1jan - 1) // 7 + 1

    return pd.DataFrame({
        "DiaSemana": dia_semana,
        "DiaMes": dt.day,
        "SemanaAno": semana,
        "Mes": dt.month,
        "TrimestreAno": dt.quarter,
        "Ano": dt.year,
    }, index=datas.index).astype(TIPO_PARTES)


def adicionar_partes_da_data(df: pd.DataFrame, datefirst: int = 7) -> tuple[pd.DataFrame, dict[str, int]]:
    """Cria as colunas que faltam; nas que já vieram da exportação, só preenche os nulos
    (dias adicionados pelo calendário).

    Retorna (df, divergências por coluna entre o valor exportado e o calculado).
    """
    calculadas = partes_da_data(df["Data"], datefirst)
    divergencias = {}
    for coluna in calculadas.columns:
        if coluna in df.columns:
            existentes = df[coluna]
            divergencias[coluna] = int((existentes.notna() & (existentes != calculadas[coluna])).sum())
            df[coluna] = existentes.fillna(calculadas[coluna]).astype(TIPO_PARTES)
        else:
            df[coluna] = calculadas[coluna]
    return df, divergencias
