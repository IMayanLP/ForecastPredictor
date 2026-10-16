"""Partes da data × valores do DATEPART do SQL Server.

Rodar: .venv/Scripts/python.exe tests/test_partes_data.py  (ou pytest, se instalado)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddmrp_forecast.features.datas import adicionar_partes_da_data, partes_da_data  # noqa: E402

# data → (WEEKDAY, WEEK) no SQL Server com SET DATEFIRST 7
ESPERADO_DATEFIRST_7 = {
    "2026-01-01": (5, 1),    # quinta
    "2026-01-03": (7, 1),    # sábado, ainda semana 1
    "2026-01-04": (1, 2),    # domingo abre a semana 2
    "2024-02-29": (5, 9),
    "2024-12-31": (3, 53),
    "2022-12-31": (7, 53),
    "2023-01-01": (1, 1),    # 1º/jan num domingo
    "2023-01-08": (1, 2),
}


def test_datefirst_7():
    datas = pd.Series(pd.to_datetime(list(ESPERADO_DATEFIRST_7)))
    partes = partes_da_data(datas, 7)
    for i, (data, (dia_semana, semana)) in enumerate(ESPERADO_DATEFIRST_7.items()):
        assert partes["DiaSemana"].iloc[i] == dia_semana, (data, "DiaSemana", partes["DiaSemana"].iloc[i])
        assert partes["SemanaAno"].iloc[i] == semana, (data, "SemanaAno", partes["SemanaAno"].iloc[i])
    linha = partes.iloc[3]   # 2024-02-29
    assert (linha["DiaMes"], linha["Mes"], linha["TrimestreAno"], linha["Ano"]) == (29, 2, 1, 2024)


def test_datefirst_1():
    datas = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-04", "2026-01-05"]))
    partes = partes_da_data(datas, 1)
    assert partes["DiaSemana"].tolist() == [4, 7, 1]
    assert partes["SemanaAno"].tolist() == [1, 1, 2]


def test_preenche_so_o_que_falta():
    df = pd.DataFrame({
        "Data": pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-05"]),
        "DiaSemana": [7.0, np.nan, 2.0],        # dia do meio = adicionado pelo calendário
    })
    df, divergencias = adicionar_partes_da_data(df, 7)
    assert df["DiaSemana"].tolist() == [7, 1, 2] and str(df["DiaSemana"].dtype) == "int16"
    assert divergencias == {"DiaSemana": 0}
    assert {"DiaMes", "SemanaAno", "Mes", "TrimestreAno", "Ano"} <= set(df.columns)

    df["DiaSemana"] = df["DiaSemana"].astype(float)
    df.loc[0, "DiaSemana"] = 6.0                # exportado com outro DATEFIRST
    _, divergencias = adicionar_partes_da_data(df, 7)
    assert divergencias["DiaSemana"] == 1


if __name__ == "__main__":
    test_datefirst_7()
    test_datefirst_1()
    test_preenche_so_o_que_falta()
    print("Partes da data: OK")
