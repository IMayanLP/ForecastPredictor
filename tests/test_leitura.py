# Leitura só de séries com venda desde DATA_INICIO (em lotes) e consumo negativo → 0.
# Rodar: .venv/Scripts/python.exe tests/test_leitura.py  (ou pytest, se instalado)

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddmrp_forecast.config.settings import CHAVE_SERIE  # noqa: E402
from ddmrp_forecast.io.parquet_store import ler_series_com_venda  # noqa: E402
from ddmrp_forecast.preprocess.preparacao import zerar_consumos_negativos  # noqa: E402


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


def test_zerar_consumos_negativos():
    df = pd.DataFrame({"Consumo": [3.0, -1.0, 0.0, -0.5, None]})
    df, zerados = zerar_consumos_negativos(df)
    assert zerados == 2
    assert df["Consumo"].tolist()[:4] == [3.0, 0.0, 0.0, 0.0] and pd.isna(df["Consumo"].iloc[4])


if __name__ == "__main__":
    test_so_series_com_venda()
    test_zerar_consumos_negativos()
    print("Leitura: OK")
