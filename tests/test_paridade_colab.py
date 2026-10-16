"""Compara o pipeline modular com o código original do Colab em dados sintéticos.

Rodar: .venv/Scripts/python.exe tests/test_paridade_colab.py  (ou pytest, se instalado)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddmrp_forecast.features.defasagens import criar_lags, criar_medias_moveis  # noqa: E402
from ddmrp_forecast.preprocess.adu import recalcular_adu  # noqa: E402
from ddmrp_forecast.preprocess.calendario import completar_calendario  # noqa: E402
from ddmrp_forecast.preprocess.picos import limpar_picos, marcar_picos  # noqa: E402
from ddmrp_forecast.preprocess.preparacao import preparar_base  # noqa: E402

CHAVE = ("Cliente", "IdBodega", "Referencia")
COLUNAS_FIXAS = ["DiasADU", "ADI", "CV", "TiempoReposicion", "Descripcion", "Unidad"]
LAGS = (1, 2, 3, 7, 14, 28, 365)
JANELAS = (7, 28, 90, 180, 365)


def gerar_base(semente: int = 0) -> pd.DataFrame:
    """Séries com buracos, picos, mestre/componente, DiasADU nulo/zero/fracionário."""
    rng = np.random.default_rng(semente)
    series = [
        ("B1", "A", 7), ("B1", "#A", 7), ("B2", "A", 14),   # A some só com escopo global
        ("B1", "C", np.nan), ("B1", "D", 0), ("B2", "E", 7.9),
        ("B3", "F", 30), ("B3", "G", 7), ("B3", "H", 1),
    ]
    linhas = []
    for i, (bodega, ref, dias_adu) in enumerate(series):
        inicio = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 40)))
        datas = pd.date_range(inicio, periods=420 + 13 * i, freq="D")
        datas = datas[rng.random(len(datas)) > 0.2]          # ~20% de dias sem registro
        consumo = rng.poisson(5, len(datas)).astype(float)
        consumo[rng.random(len(datas)) < 0.03] *= 20          # picos
        if i == 3:
            consumo[0] = 500                                  # pico logo no 1º registro
        for data, valor in zip(datas, consumo):
            linhas.append({
                "Data": data, "Cliente": "CLI", "IdBodega": bodega, "Referencia": ref,
                "Consumo": valor, "DiasADU": dias_adu if rng.random() > 0.1 else np.nan,
                "ADI": 1.5, "CV": 0.4, "TiempoReposicion": 10.0,
                "Descripcion": f"Item {ref}", "Unidad": "PC",
            })
    df = pd.DataFrame(linhas)
    for coluna in ["Cliente", "IdBodega", "Referencia", "Descripcion", "Unidad"]:
        df[coluna] = df[coluna].astype("category")   # como sai de ler_parquet
    return df


def colab(df: pd.DataFrame) -> pd.DataFrame:
    """Código do Colab, adaptado só onde o pandas 3 quebraria o original (marcado com ADAPTADO)."""
    df = df.copy()
    for coluna in ["Cliente", "IdBodega", "Referencia", "Descripcion", "Unidad"]:
        df[coluna] = df[coluna].astype(str)
    df["Unidad"] = df["Unidad"].str.strip()

    buffers_mestre = set(df.loc[df["Referencia"].str.startswith("#", na=False), "Referencia"])
    filhos_BM = {ref[1:] for ref in buffers_mestre}
    df = df[~(df["Referencia"].isin(filhos_BM) & ~df["Referencia"].str.startswith("#", na=False))].copy()
    df = df.sort_values(["IdBodega", "Referencia", "Data"]).reset_index(drop=True)

    quartis = (
        df.groupby(["IdBodega", "Referencia"])["Consumo"]
        .quantile([0.25, 0.75]).unstack()
        .rename(columns={0.25: "Q1", 0.75: "Q3"}).reset_index()
    )
    quartis["IQR"] = quartis["Q3"] - quartis["Q1"]
    quartis["LimiteSuperior"] = quartis["Q3"] + 3 * quartis["IQR"]
    df = df.merge(quartis, on=["IdBodega", "Referencia"], how="left")
    df["EhPico"] = df["Consumo"] > df["LimiteSuperior"]

    df["Consumo_Limpo"] = df["Consumo"].where(~df["EhPico"])
    grupo = df.groupby(["IdBodega", "Referencia"])["Consumo_Limpo"]
    df["Media_7_Limpa"] = grupo.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df.loc[df["EhPico"], "Consumo_Limpo"] = df.loc[df["EhPico"], "Media_7_Limpa"]

    chaves = ["IdBodega", "Referencia"]
    # ADAPTADO: apply(axis=1) + explode trocado por loop (mesmo resultado, sem ambiguidade do apply)
    limites = df.groupby(chaves)["Data"].agg(["min", "max"])
    datas_completas = pd.concat([
        pd.DataFrame({"IdBodega": b, "Referencia": r, "Data": pd.date_range(lin["min"], lin["max"], freq="D")})
        for (b, r), lin in limites.iterrows()
    ], ignore_index=True)
    df_completo = datas_completas.merge(df, on=chaves + ["Data"], how="left")
    df_completo["Consumo"] = df_completo["Consumo"].fillna(0)
    df_completo["EhPico"] = df_completo["EhPico"].astype("boolean").fillna(False).astype(bool)
    df_completo["Consumo_Limpo"] = df_completo["Consumo_Limpo"].fillna(0)
    df_completo = df_completo.sort_values(["IdBodega", "Referencia", "Data"]).reset_index(drop=True)
    colunas_fixas = ["Cliente", "DiasADU", "ADI", "CV", "TiempoReposicion", "Descripcion", "Unidad"]
    df_completo[colunas_fixas] = (
        df_completo.groupby(["IdBodega", "Referencia"])[colunas_fixas].transform(lambda x: x.ffill().bfill())
    )

    # ADAPTADO: groupby.apply do pandas 3 descarta as colunas de grupo; mesmo cálculo por loop
    df_completo["ADU_recalculado"] = np.nan
    for _, idx in df_completo.groupby(chaves).groups.items():
        g = df_completo.loc[idx].sort_values("Data")
        dias_adu = g["DiasADU"].dropna()
        if dias_adu.empty:
            continue
        dias_adu = int(dias_adu.iloc[0])
        if dias_adu <= 0:
            continue
        df_completo.loc[g.index, "ADU_recalculado"] = (
            g["Consumo"].shift(1).rolling(dias_adu, min_periods=dias_adu).mean()
        )

    grupo = df_completo.groupby(["IdBodega", "Referencia"])["Consumo_Limpo"]
    for lag in LAGS:
        df_completo[f"Lag_{lag}"] = grupo.shift(lag)
    for janela in JANELAS:
        df_completo[f"Media_{janela}"] = grupo.transform(
            lambda x, j=janela: x.shift(1).rolling(j, min_periods=j).mean()
        )
    return df_completo


def modular(df: pd.DataFrame) -> pd.DataFrame:
    df, _ = preparar_base(df.copy(), escopo_buffer_mestre=())   # escopo global = Colab
    df, _ = marcar_picos(df, CHAVE, 3.0, quartis_so_com_consumo=False)   # regra do Colab
    df = limpar_picos(df, CHAVE, 7)
    df, _ = completar_calendario(df, CHAVE, COLUNAS_FIXAS)
    df = recalcular_adu(df, CHAVE)
    df = criar_lags(df, CHAVE, LAGS)
    return criar_medias_moveis(df, CHAVE, JANELAS)


def test_paridade_colab():
    base = gerar_base()
    esperado = colab(base)
    obtido = modular(base)

    assert len(obtido) == len(esperado), (len(obtido), len(esperado))
    ordem = ["IdBodega", "Referencia", "Data"]
    esperado = esperado.sort_values(ordem).reset_index(drop=True)
    obtido = obtido.sort_values(ordem).reset_index(drop=True)

    for coluna in ["IdBodega", "Referencia", "Cliente", "Descripcion", "Unidad"]:
        assert (obtido[coluna].astype(str) == esperado[coluna].astype(str)).all(), coluna
    assert (obtido["Data"].to_numpy() == esperado["Data"].to_numpy()).all(), "Data"
    assert (obtido["EhPico"].to_numpy() == esperado["EhPico"].to_numpy()).all(), "EhPico"

    numericas = ["Consumo", "LimiteSuperior", "Consumo_Limpo", "Media_7_Limpa", "DiasADU", "ADI", "CV",
                 "TiempoReposicion", "ADU_recalculado",
                 *[f"Lag_{l}" for l in LAGS], *[f"Media_{j}" for j in JANELAS]]
    for coluna in numericas:
        np.testing.assert_allclose(obtido[coluna].to_numpy(float), esperado[coluna].to_numpy(float),
                                   rtol=1e-6, equal_nan=True, err_msg=coluna)
        print(f"  ok {coluna:<18} nulos={int(obtido[coluna].isna().sum()):>5}")


def test_escopo_buffer_mestre():
    # com escopo (Cliente, IdBodega), "A" da B2 fica (não há "#A" na B2)
    base = gerar_base()
    df, removidas = preparar_base(base.copy(), ("Cliente", "IdBodega"))
    refs = set(zip(df["IdBodega"].astype(str), df["Referencia"].astype(str)))
    assert ("B1", "A") not in refs and ("B2", "A") in refs and ("B1", "#A") in refs
    assert removidas == int(((base["IdBodega"] == "B1") & (base["Referencia"] == "A")).sum())


if __name__ == "__main__":
    test_paridade_colab()
    test_escopo_buffer_mestre()
    print("Paridade com o Colab: OK")
