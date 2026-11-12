# Leitura só de séries com venda desde DATA_INICIO (em lotes) e consumo negativo → 0.
# Rodar: .venv/Scripts/python.exe tests/test_leitura.py  (ou pytest, se instalado)

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddmrp_forecast.config.settings import CHAVE_SERIE  # noqa: E402
from ddmrp_forecast.io.parquet_store import ler_series_com_venda, sortear_series  # noqa: E402
from ddmrp_forecast.io.selecao import filtrar_por_cliente  # noqa: E402
from ddmrp_forecast.preprocess.preparacao import (  # noqa: E402
    remover_duplicados_identicos, verificar_ordenacao, zerar_consumos_negativos,
)


def escrever(pasta: Path, nome: str, linhas: list[tuple]) -> None:
    df = pd.DataFrame(linhas, columns=["Referencia", "Data", "Consumo"]).assign(Cliente="CERAMFIX", IdBodega="1")
    df["Data"] = pd.to_datetime(df["Data"]).astype("datetime64[ms]")
    # chaves como string (não large_string), igual ao Parquet gerado pela conversão
    tabela = pa.Table.from_pandas(df, preserve_index=False)
    tabela = tabela.cast(pa.schema([pa.field(c.name, pa.string()) if pa.types.is_large_string(c.type) or
                                    pa.types.is_string(c.type) else c for c in tabela.schema]))
    pq.write_table(tabela, pasta / nome)


def test_so_series_com_venda():
    with tempfile.TemporaryDirectory() as temporaria:
        pasta = Path(temporaria)
        escrever(pasta, "ceramfix_2021.parquet", [
            ("VENDE", "2021-12-30", 5.0), ("SO_ANTIGA", "2021-06-01", 9.0), ("SO_ANTIGA", "2022-01-05", 0.0),
        ])
        escrever(pasta, "ceramfix_2022.parquet", [
            ("VENDE", "2022-01-01", 0.0), ("VENDE", "2022-01-02", 3.0), ("VENDE", "2022-01-03", -1.0),
            ("NUNCA", "2022-01-01", 0.0), ("NUNCA", "2022-01-02", 0.0),
            ("#MESTRE", "2022-01-01", 0.0), ("MESTRE", "2022-01-01", 4.0),
        ])
        escrever(pasta, "kasvi_2022.parquet", [("OUTRO", "2022-01-01", 1.0)])

        df, resumo = ler_series_com_venda(pasta, ("ceramfix",), CHAVE_SERIE, "2022-01-01", 1, tamanho_lote=2)
        # SO_ANTIGA só vendeu antes do início; NUNCA não vende; o mestre entra mesmo sem venda
        assert sorted(df["Referencia"].unique()) == ["#MESTRE", "MESTRE", "VENDE"]
        assert df["Data"].min() == pd.Timestamp("2022-01-01")
        assert len(df) == 5
        assert isinstance(df["Referencia"].dtype, pd.CategoricalDtype)
        assert resumo["séries no período"] == 5 and resumo["séries mantidas"] == 3
        assert resumo["linhas no período"] == 8 and resumo["linhas mantidas"] == 5

        df, _ = ler_series_com_venda(pasta, ("ceramfix",), CHAVE_SERIE, None, 2)
        assert sorted(df["Referencia"].unique()) == ["#MESTRE", "VENDE"]      # 2 vendas: 2021-12-30 e 2022-01-02


def test_cliente_pela_pasta():
    # data/source/<Cliente>_Consumos_.../arquivo: o nome do arquivo pode não ter o cliente
    with tempfile.TemporaryDirectory() as temporaria:
        pasta = Path(temporaria)
        (pasta / "Hennings_Consumos_2019_2026").mkdir()
        (pasta / "Ceramfix_Consumos_2009_2026" / "consumos_2026").mkdir(parents=True)
        escrever(pasta / "Hennings_Consumos_2019_2026", "Consumos_Abril_Junho_2026.parquet", [("H1", "2026-04-01", 2.0)])
        escrever(pasta / "Ceramfix_Consumos_2009_2026" / "consumos_2026", "ceramfix_consumos_2026.parquet",
                 [("C1", "2026-04-01", 3.0)])
        escrever(pasta, "kasvi_consumos_completo.parquet", [("K1", "2026-04-01", 1.0)])
        arquivos = sorted(pasta.rglob("*.parquet"))
        nomes = lambda clientes: sorted(a.stem for a in filtrar_por_cliente(arquivos, clientes, pasta))
        assert nomes(("hennings",)) == ["Consumos_Abril_Junho_2026"]
        assert nomes("ceramfix") == ["ceramfix_consumos_2026"]
        assert nomes(("Ceramfix", "HENNINGS")) == ["Consumos_Abril_Junho_2026", "ceramfix_consumos_2026"]
        assert nomes(("kasvi",)) == ["kasvi_consumos_completo"]
        # o nome da pasta temporária (fora da raiz) não conta
        assert filtrar_por_cliente(arquivos, (pasta.name.lower(),), pasta) == []
        df, _ = ler_series_com_venda(pasta, ("hennings", "ceramfix"), CHAVE_SERIE, None, 1)
        assert sorted(df["Referencia"].unique()) == ["C1", "H1"]


def test_amostra_de_series():
    series = pd.DataFrame({"Cliente": ["Hennings"] * 2000 + ["CERAMFIX"] * 50, "Referencia": [f"R{i}" for i in range(2050)],
                           "IdBodega": "1"})
    sempre = np.zeros(len(series), bool)
    sempre[0] = True    # buffer mestre entra sempre
    amostra = sortear_series(series, CHAVE_SERIE, {"hennings": 0.1}, sempre)
    hennings = amostra[amostra["Cliente"] == "Hennings"]
    assert 150 <= len(hennings) <= 250 and "R0" in set(hennings["Referencia"])
    assert (amostra["Cliente"] == "CERAMFIX").sum() == 50          # cliente fora do dicionário: todas
    # o sorteio não depende das outras séries nem da ordem
    so_hennings = sortear_series(series.iloc[:2000].iloc[::-1], CHAVE_SERIE, {"hennings": 0.1}, sempre[:2000][::-1])
    assert set(so_hennings["Referencia"]) == set(hennings["Referencia"])
    maior = sortear_series(series, CHAVE_SERIE, {"hennings": 0.2}, sempre)
    assert set(hennings["Referencia"]) <= set(maior["Referencia"])  # 10% ⊂ 20%


def test_duplicados_identicos():
    df = pd.DataFrame({"Cliente": "H", "IdBodega": "1", "Referencia": ["A", "A", "A", "B", "B"],
                       "Data": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-01", "2026-01-02"]),
                       "Consumo": [1.0, 5.0, 5.0, 2.0, 3.0]})
    limpo, removidos = remover_duplicados_identicos(df, CHAVE_SERIE)
    assert removidos == 1 and len(limpo) == 4 and limpo["Consumo"].sum() == 11.0
    verificar_ordenacao(limpo, CHAVE_SERIE)
    # mesma série e Data com consumo diferente não é removida: continua erro
    diferente = df.assign(Consumo=[1.0, 5.0, 6.0, 2.0, 3.0])
    _, removidos = remover_duplicados_identicos(diferente, CHAVE_SERIE)
    assert removidos == 0
    try:
        verificar_ordenacao(diferente, CHAVE_SERIE)
        raise AssertionError("repetição com valores diferentes deveria falhar")
    except ValueError:
        pass


def test_zerar_consumos_negativos():
    df = pd.DataFrame({"Consumo": [3.0, -1.0, 0.0, -0.5, None]})
    df, zerados = zerar_consumos_negativos(df)
    assert zerados == 2
    assert df["Consumo"].tolist()[:4] == [3.0, 0.0, 0.0, 0.0] and pd.isna(df["Consumo"].iloc[4])


if __name__ == "__main__":
    test_so_series_com_venda()
    test_cliente_pela_pasta()
    test_amostra_de_series()
    test_duplicados_identicos()
    test_zerar_consumos_negativos()
    print("Leitura: OK")
