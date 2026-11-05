import numpy as np
import pandas as pd


def metricas_por_serie(df: pd.DataFrame, chave: tuple[str, ...], previsoes: tuple[str, ...],
                       alvo: str = "Alvo") -> pd.DataFrame:
    real = df[alvo].to_numpy(float)
    tabelas = []
    for coluna in previsoes:
        previsto = df[coluna].to_numpy(float)
        parcial = pd.DataFrame({
            "Real": real, "Previsto": previsto, "ErroAbs": np.abs(real - previsto),
            "ErroQuad": (real - previsto) ** 2,
        }, index=df.index)
        soma = parcial.groupby([df[c] for c in chave], observed=True, sort=False).agg(
            Linhas=("Real", "size"), SomaReal=("Real", "sum"), SomaPrevisto=("Previsto", "sum"),
            SomaErroAbs=("ErroAbs", "sum"), SomaErroQuad=("ErroQuad", "sum"),
        )
        soma["MAE"] = soma["SomaErroAbs"] / soma["Linhas"]
        soma["RMSE"] = np.sqrt(soma["SomaErroQuad"] / soma["Linhas"])
        soma["WAPE"] = (soma["SomaErroAbs"] / soma["SomaReal"]).where(soma["SomaReal"] > 0)
        soma["Vies"] = (soma["SomaPrevisto"] / soma["SomaReal"] - 1).where(soma["SomaReal"] > 0)
        tabelas.append(soma.assign(Modelo=coluna).reset_index())
    return pd.concat(tabelas, ignore_index=True)


def comparar_modelos(por_serie: pd.DataFrame, chave: tuple[str, ...], referencia: str) -> pd.DataFrame:
    total = por_serie.groupby("Modelo", sort=False).agg(
        Linhas=("Linhas", "sum"), SomaReal=("SomaReal", "sum"), SomaPrevisto=("SomaPrevisto", "sum"),
        SomaErroAbs=("SomaErroAbs", "sum"), SomaErroQuad=("SomaErroQuad", "sum"),
        WAPEMedianoSerie=("WAPE", "median"),
    )
    total["MAE"] = total["SomaErroAbs"] / total["Linhas"]
    total["RMSE"] = np.sqrt(total["SomaErroQuad"] / total["Linhas"])
    total["WAPE"] = total["SomaErroAbs"] / total["SomaReal"]
    total["Vies"] = total["SomaPrevisto"] / total["SomaReal"] - 1

    # % de séries (com venda) em que a referência erra menos que cada concorrente
    wape = por_serie.pivot_table(index=list(chave), columns="Modelo", values="WAPE", observed=True)
    total[f"{referencia}Vence%"] = [
        (wape[referencia] < wape[m]).sum() / wape[[referencia, m]].notna().all(axis=1).sum() * 100
        if m != referencia else np.nan
        for m in total.index
    ]
    return total[["Linhas", "MAE", "RMSE", "WAPE", "Vies", "WAPEMedianoSerie", f"{referencia}Vence%"]]


def segmentar_abc(df: pd.DataFrame, chave: tuple[str, ...], coluna: str, linhas: np.ndarray,
                  cortes: tuple[float, float] = (0.8, 0.95)) -> pd.Series:
    # A = séries que formam os primeiros 80% do volume, B até 95%, C o resto (inclusive sem venda)
    volume = df.loc[linhas].groupby(list(chave), observed=True, sort=False)[coluna].sum().sort_values(ascending=False)
    antes = (volume.cumsum() / volume.sum()).shift(fill_value=0).fillna(1)
    segmento = np.select([antes < cortes[0], antes < cortes[1]], ["A", "B"], "C")
    return pd.Series(segmento, index=volume.index, name="Segmento")


def medianas_por_segmento(por_serie: pd.DataFrame, chave: tuple[str, ...], segmentos: pd.Series) -> dict[str, float]:
    # WAPE mediano por série dentro de cada segmento; série fora da segmentação conta como C
    segmento = por_serie.join(segmentos, on=list(chave))["Segmento"].fillna("C")
    return por_serie.groupby(segmento.to_numpy())["WAPE"].median().to_dict()


def comparar_por_segmento(por_serie: pd.DataFrame, chave: tuple[str, ...], segmentos: pd.Series) -> pd.DataFrame:
    # WAPE mediano por série de cada modelo dentro de A/B/C (+ quantas séries com venda em cada segmento)
    por_serie = por_serie.reset_index(drop=True)
    por_serie["Segmento"] = por_serie.join(segmentos, on=list(chave))["Segmento"].fillna("C")
    tabela = por_serie.groupby(["Modelo", "Segmento"], sort=False)["WAPE"].median().unstack()
    series = por_serie[(por_serie["Modelo"] == por_serie["Modelo"].iloc[0]) & por_serie["WAPE"].notna()]
    contagem = series.groupby("Segmento").size()
    tabela = tabela.reindex(columns=sorted(tabela.columns))
    tabela.columns = [f"WAPEMediano{s} ({contagem.get(s, 0):,} séries)" for s in tabela.columns]
    return tabela
