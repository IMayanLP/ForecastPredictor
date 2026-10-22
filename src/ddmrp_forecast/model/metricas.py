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
