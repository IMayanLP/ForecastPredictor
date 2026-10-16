"""DiasADU 0 → padrão; auditoria de picos e ADU detecta alterações; ordem alfabética das chaves.

Rodar: .venv/Scripts/python.exe tests/test_auditoria.py  (ou pytest, se instalado)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_paridade_colab import CHAVE, gerar_base, modular  # noqa: E402

from ddmrp_forecast.config.settings import CHAVE_SERIE  # noqa: E402
from ddmrp_forecast.preprocess.adu import corrigir_dias_adu  # noqa: E402
from ddmrp_forecast.preprocess.auditoria import auditar_adu, auditar_picos  # noqa: E402
from ddmrp_forecast.preprocess.picos import marcar_picos  # noqa: E402
from ddmrp_forecast.preprocess.preparacao import preparar_base, verificar_ordenacao  # noqa: E402


def test_corrigir_dias_adu():
    df = pd.DataFrame({"DiasADU": [0.0, 7.0, np.nan, 0.0]})
    df, corrigidos = corrigir_dias_adu(df, 30.0)
    assert corrigidos == 2
    assert df["DiasADU"].tolist()[:2] == [30.0, 7.0] and np.isnan(df["DiasADU"].iloc[2])


def test_auditoria_detecta_alteracoes():
    df = modular(gerar_base())
    assert sum(auditar_picos(df)[1].values()) == 0
    assert auditar_adu(df, CHAVE)[1]["ADU_recalculado diferente do esperado"] == 0
    assert auditar_adu(df, CHAVE)[1]["linhas com DiasADU = 0"] > 0   # série "D" da base sintética

    linha_adu = df["ADU_recalculado"].first_valid_index()
    df.loc[linha_adu, "ADU_recalculado"] += 0.01
    assert auditar_adu(df, CHAVE)[1]["ADU_recalculado diferente do esperado"] == 1

    linha_pico = df.index[df["EhPico"]][0]
    df.loc[linha_pico, "Consumo_Limpo"] = df.loc[linha_pico, "Consumo"]
    assert auditar_picos(df)[1]["Consumo_Limpo diferente do esperado"] == 1


def test_ordem_alfabetica_com_category():
    # categorias fora da ordem alfabética, como saem do Parquet
    df = pd.DataFrame({
        "Cliente": ["K", "K", "K", "A"],
        "Referencia": ["R2", "R1", "R1", "R9"],
        "IdBodega": ["1", "2", "1", "1"],
        "Data": pd.to_datetime(["2026-01-01"] * 4),
    })
    for coluna in CHAVE_SERIE:
        df[coluna] = pd.Categorical(df[coluna], categories=list(dict.fromkeys(df[coluna])))
    df, _ = preparar_base(df, ())
    verificar_ordenacao(df, CHAVE_SERIE)
    assert list(zip(*(df[c].astype(str) for c in CHAVE_SERIE))) == [
        ("A", "R9", "1"), ("K", "R1", "1"), ("K", "R1", "2"), ("K", "R2", "1")]


def test_quartis_so_com_consumo():
    # 80% de dias sem venda (Q3 = 0 com os zeros); vendas 5, 6, 5, 7, 6, 80
    consumo = [0.0] * 30
    for posicao, valor in zip(range(2, 30, 5), [5, 6, 5, 7, 6, 80]):
        consumo[posicao] = float(valor)
    df = pd.DataFrame({
        "Cliente": "K", "Referencia": ["R1"] * len(consumo) + ["R2"] * 3, "IdBodega": "1",
        "Consumo": consumo + [0.0, 0.0, 0.0],
    })
    com_zeros, _ = marcar_picos(df.copy(), CHAVE_SERIE, 3.0, quartis_so_com_consumo=False)
    assert com_zeros.loc[:29, "EhPico"].sum() == 6          # Colab: toda venda vira pico

    so_vendas, quartis = marcar_picos(df.copy(), CHAVE_SERIE, 3.0, quartis_so_com_consumo=True)
    assert so_vendas.index[so_vendas["EhPico"]].tolist() == [27]   # só o 80
    assert quartis.loc[0, ["Q1", "Q3", "LimiteSuperior"]].tolist() == [5.25, 6.75, 11.25]
    assert so_vendas.loc[30:, "LimiteSuperior"].isna().all() and not so_vendas.loc[30:, "EhPico"].any()


if __name__ == "__main__":
    test_corrigir_dias_adu()
    test_auditoria_detecta_alteracoes()
    test_ordem_alfabetica_com_category()
    test_quartis_so_com_consumo()
    print("Auditoria: OK")
