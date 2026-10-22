import numpy as np
import pandas as pd

UM_DIA = np.timedelta64(1, "D")


def datas_corte(data_fim: pd.Timestamp, dias_teste: int, dias_validacao: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    inicio_teste = pd.Timestamp(data_fim) - pd.Timedelta(days=dias_teste - 1)
    inicio_validacao = inicio_teste - pd.Timedelta(days=dias_validacao)
    return inicio_validacao, inicio_teste


def separar_bases(df: pd.DataFrame, horizonte: int, inicio_validacao: pd.Timestamp,
                  inicio_teste: pd.Timestamp, alvo: str = "Alvo") -> pd.Series:
    # a janela do alvo [Data, Data + horizonte - 1] não pode atravessar a fronteira entre bases
    inicio = df["Data"]
    fim = inicio + (horizonte - 1) * UM_DIA
    tem_alvo = df[alvo].notna()
    base = pd.Series(pd.NA, index=df.index, dtype="string")
    base[tem_alvo & (fim < inicio_validacao)] = "treino"
    base[tem_alvo & (inicio >= inicio_validacao) & (fim < inicio_teste)] = "validacao"
    base[tem_alvo & (inicio >= inicio_teste)] = "teste"
    return base.astype("category")


def resumir_bases(df: pd.DataFrame, base: pd.Series, alvo: str = "Alvo") -> pd.DataFrame:
    return (
        df.groupby(base, observed=True)
        .agg(Linhas=("Data", "size"), Inicio=("Data", "min"), Fim=("Data", "max"),
             AlvoMedio=(alvo, "mean"), AlvoZero=(alvo, lambda s: (s == 0).mean()))
        .reindex(["treino", "validacao", "teste"])
    )
