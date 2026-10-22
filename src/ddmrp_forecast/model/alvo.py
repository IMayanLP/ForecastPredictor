import pandas as pd

from ddmrp_forecast.features.janelas import agrupar_series, media_movel_por_codigo


def criar_alvo(df: pd.DataFrame, chave: tuple[str, ...], coluna: str, horizonte: int,
               nome: str = "Alvo") -> pd.DataFrame:
    # média em [t, t + horizonte - 1]: a média móvel termina em t + horizonte - 1 e volta para t
    # NaN quando a série acaba antes de completar a janela
    codigos = agrupar_series(df, chave).ngroup()
    media = media_movel_por_codigo(df[coluna], codigos, horizonte, horizonte)
    df[nome] = media.groupby(codigos, sort=False).shift(-(horizonte - 1))
    return df
