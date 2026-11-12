import numpy as np
import pandas as pd

UM_DIA = np.timedelta64(1, "D")


def datas_corte(data_fim: pd.Timestamp, dias_teste: int, dias_validacao: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    inicio_teste = pd.Timestamp(data_fim) - pd.Timedelta(days=dias_teste - 1)
    inicio_validacao = inicio_teste - pd.Timedelta(days=dias_validacao)
    return inicio_validacao, inicio_teste


def datas_corte_por_cliente(df: pd.DataFrame, dias_teste: int, dias_validacao: int,
                            por: str = "Cliente") -> pd.DataFrame:
    # cada cliente tem o seu teste nos últimos dias da própria exportação (as exportações terminam em datas diferentes)
    fim = df.groupby(por, observed=True)["Data"].max()
    cortes = [datas_corte(f, dias_teste, dias_validacao) for f in fim]
    return pd.DataFrame(cortes, index=fim.index, columns=["InicioValidacao", "InicioTeste"]).assign(Fim=fim)


def por_linha(df: pd.DataFrame, valores: pd.Series, por: str = "Cliente") -> np.ndarray:
    # valor do cliente de cada linha (ex.: data de corte); cliente sem valor → NaT
    coluna = df[por]
    if isinstance(coluna.dtype, pd.CategoricalDtype):
        tabela = pd.to_datetime(valores.reindex(coluna.cat.categories)).to_numpy("datetime64[ns]")
        return np.append(tabela, np.datetime64("NaT", "ns"))[coluna.cat.codes.to_numpy()]
    return pd.to_datetime(coluna.map(valores)).to_numpy("datetime64[ns]")


def inicio_da_base(df: pd.DataFrame, linhas: np.ndarray, por: str = "Cliente") -> np.ndarray:
    # 1ª data das linhas marcadas (ex.: do teste) dentro do cliente de cada linha do df
    return por_linha(df, df.loc[linhas, [por, "Data"]].groupby(por, observed=True)["Data"].min(), por)


def separar_bases(df: pd.DataFrame, horizonte: int, inicio_validacao: pd.Timestamp | np.ndarray,
                  inicio_teste: pd.Timestamp | np.ndarray, alvo: str = "Alvo") -> pd.Series:
    # cortes únicos (Timestamp) ou um por linha (por_linha), para clientes com exportações de datas diferentes
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
