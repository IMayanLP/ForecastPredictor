"""Etapa 3 — detecção e limpeza de picos de consumo (IQR por série)."""

import pandas as pd

from ddmrp_forecast.features.janelas import agrupar_series, media_movel_defasada


def marcar_picos(df: pd.DataFrame, chave: tuple[str, ...], fator_iqr: float = 3.0,
                 quartis_so_com_consumo: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adiciona LimiteSuperior e EhPico. Retorna (df, quartis por série).

    Q1/Q3/IQR ficam só na tabela de quartis (no Colab iam para cada linha via merge).
    """
    grupos = agrupar_series(df, chave)
    codigos = grupos.ngroup().to_numpy()

    # séries intermitentes: com os zeros, Q1 = Q3 = 0 e qualquer venda vira pico
    consumo = df["Consumo"].where(df["Consumo"] > 0) if quartis_so_com_consumo else df["Consumo"]
    por_serie = consumo.groupby(codigos)
    quartis = pd.DataFrame({
        "Q1": por_serie.quantile(0.25).to_numpy(),
        "Q3": por_serie.quantile(0.75).to_numpy(),
    }, index=grupos.size().index)
    quartis["IQR"] = quartis["Q3"] - quartis["Q1"]
    quartis["LimiteSuperior"] = quartis["Q3"] + fator_iqr * quartis["IQR"]

    # série → linha pelo código do grupo (mesma ordem do groupby), sem merge
    df["LimiteSuperior"] = quartis["LimiteSuperior"].to_numpy()[codigos]
    df["EhPico"] = df["Consumo"] > df["LimiteSuperior"]
    return df, quartis.reset_index()


def resumir_picos(df: pd.DataFrame, chave: tuple[str, ...]) -> pd.DataFrame:
    resumo = (
        agrupar_series(df, chave)
        .agg(
            TotalDias=("Consumo", "size"),
            QuantidadePicos=("EhPico", "sum"),
            LimiteSuperior=("LimiteSuperior", "first"),
        )
        .reset_index()
    )
    resumo["PercentualPicos"] = resumo["QuantidadePicos"] / resumo["TotalDias"] * 100
    return resumo.sort_values("QuantidadePicos", ascending=False)


def limpar_picos(df: pd.DataFrame, chave: tuple[str, ...], janela: int = 7) -> pd.DataFrame:
    """Consumo_Limpo = Consumo com cada pico trocado pela média dos `janela` registros anteriores sem pico.

    A janela conta registros, não dias: roda antes de completar o calendário (igual ao Colab).
    """
    df["Consumo_Limpo"] = df["Consumo"].where(~df["EhPico"])
    coluna_media = f"Media_{janela}_Limpa"
    df[coluna_media] = media_movel_defasada(df, chave, "Consumo_Limpo", janela, min_periodos=1)
    df["Consumo_Limpo"] = df["Consumo_Limpo"].mask(df["EhPico"], df[coluna_media])
    return df
