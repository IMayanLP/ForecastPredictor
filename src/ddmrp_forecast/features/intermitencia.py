import numpy as np
import pandas as pd

from ddmrp_forecast.features.janelas import agrupar_series, media_movel_defasada


def _suavizar(valores: pd.Series, codigos: np.ndarray, alfa: float) -> pd.Series:
    # suavização exponencial por série; NaN não atualiza a média e recebe o último valor suavizado
    suavizado = valores.groupby(codigos, sort=False).ewm(alpha=alfa, adjust=False, ignore_na=True).mean()
    return suavizado.droplevel(0).sort_index()


def criar_frequencias(df: pd.DataFrame, chave: tuple[str, ...], janelas: tuple[int, ...],
                      coluna: str = "Consumo_Limpo", tipo: str = "float32") -> pd.DataFrame:
    # Freq_N = fração dos N dias anteriores com venda; TamanhoMedio_N = Media_N / Freq_N (só dias com venda)
    df["_Venda"] = (df[coluna] > 0).astype("float32")
    for janela in janelas:
        df[f"Freq_{janela}"] = media_movel_defasada(df, chave, "_Venda", janela).astype(tipo)
        if f"Media_{janela}" in df.columns:
            frequencia = df[f"Freq_{janela}"]
            df[f"TamanhoMedio_{janela}"] = (df[f"Media_{janela}"] / frequencia.where(frequencia > 0)).astype(tipo)
    del df["_Venda"]
    return df


def criar_recencia(df: pd.DataFrame, chave: tuple[str, ...], coluna: str = "Consumo_Limpo",
                   tipo: str = "float32") -> pd.DataFrame:
    # DiasHistorico = dias da série antes de t; DiasDesdeVenda = dias desde a última venda até t-1
    grupos = agrupar_series(df, chave)
    codigos = grupos.ngroup().to_numpy()
    posicao = grupos.cumcount().to_numpy()
    venda = (df[coluna] > 0).to_numpy()
    ultima_venda = pd.Series(np.where(venda, posicao, np.nan)).groupby(codigos).shift(1).groupby(codigos).ffill()
    df["DiasHistorico"] = posicao.astype(tipo)
    df["DiasDesdeVenda"] = (posicao - ultima_venda.to_numpy()).astype(tipo)
    return df


def criar_tsb_sba(df: pd.DataFrame, chave: tuple[str, ...], alfa: float = 0.1, coluna: str = "Consumo_Limpo",
                  tipo: str = "float32") -> pd.DataFrame:
    # métodos clássicos de demanda intermitente, com dados até t-1
    grupos = agrupar_series(df, chave)
    codigos = grupos.ngroup().to_numpy()
    posicao = grupos.cumcount().to_numpy()
    venda = (df[coluna] > 0).to_numpy()
    valores = df[coluna].to_numpy(float)

    # TSB: probabilidade de venda (atualizada todo dia) × tamanho (atualizado só nos dias com venda)
    indicador = pd.Series(venda.astype(float)).groupby(codigos).shift(1)
    tamanho = pd.Series(np.where(venda, valores, np.nan)).groupby(codigos).shift(1)
    tamanho_suavizado = _suavizar(tamanho, codigos, alfa).groupby(codigos).ffill()
    df["TSB"] = (_suavizar(indicador, codigos, alfa) * tamanho_suavizado).fillna(0).to_numpy().astype(tipo)

    # SBA: tamanho e intervalo entre vendas suavizados nos dias de venda; vale a partir do dia seguinte
    linhas_venda = np.flatnonzero(venda)
    codigos_venda = codigos[linhas_venda]
    intervalo = pd.Series(posicao[linhas_venda].astype(float)).groupby(codigos_venda).diff()
    intervalo = intervalo.fillna(pd.Series(posicao[linhas_venda] + 1.0))     # 1ª venda: dias desde o início
    tamanho_venda = _suavizar(pd.Series(valores[linhas_venda]), codigos_venda, alfa)
    intervalo_venda = _suavizar(intervalo, codigos_venda, alfa)
    sba = np.full(len(df), np.nan)
    sba[linhas_venda] = ((1 - alfa / 2) * tamanho_venda / intervalo_venda).to_numpy()
    df["SBA"] = pd.Series(sba).groupby(codigos).shift(1).groupby(codigos).ffill().to_numpy().astype(tipo)
    return df
