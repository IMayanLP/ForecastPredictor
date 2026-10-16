""".rpt e .csv geram o mesmo Parquet; lista branca; checagem de ordenação.

Rodar: .venv/Scripts/python.exe tests/test_conversao.py  (ou pytest, se instalado)
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddmrp_forecast.config.settings import CHAVE_SERIE  # noqa: E402
from ddmrp_forecast.io.conversao import converter_pasta, listar_fontes  # noqa: E402
from ddmrp_forecast.io.parquet_store import ler_parquet  # noqa: E402
from ddmrp_forecast.preprocess.preparacao import preparar_base, verificar_ordenacao  # noqa: E402

LINHAS = [
    ["Cliente", "Data", "IdBodega", "Referencia", "Consumo", "DiasADU", "Unidad"],
    ["KASVI", "2026-01-02 00:00:00.000", "2", "I01", "3.5000", "180.0000", "UN"],
    ["KASVI", "2026-01-01 00:00:00.000", "2", "I01", "0.0000", "NULL", "UN"],
    ["KASVI", "2026-01-01 00:00:00.000", "1", "#I01", "12.2500", "30.0000", "PC"],
]


def escrever_rpt(caminho: Path) -> None:
    larguras = [max(len(l[i]) for l in LINHAS) + 2 for i in range(len(LINHAS[0]))]
    formatar = lambda l: " ".join(v.ljust(w) for v, w in zip(l, larguras))  # noqa: E731
    texto = [formatar(LINHAS[0]), " ".join("-" * w for w in larguras), *map(formatar, LINHAS[1:]),
             "", f"({len(LINHAS) - 1} rows affected)"]
    caminho.write_bytes(("\r\n".join(texto) + "\r\n").encode("utf-8-sig"))


def escrever_csv(caminho: Path, separador: str = ",") -> None:
    linhas = [[v.replace(".", ",") if separador == ";" and i in (4, 5) else v for i, v in enumerate(l)]
              for l in LINHAS]
    linhas = [["" if v == "NULL" else v for v in l] for l in linhas]
    caminho.write_bytes(("\n".join(separador.join(l) for l in linhas) + "\n").encode("utf-8-sig"))


def test_rpt_e_csv_geram_o_mesmo_parquet():
    with tempfile.TemporaryDirectory() as tmp:
        origem, destino = Path(tmp, "src"), Path(tmp, "raw")
        origem.mkdir()
        escrever_rpt(origem / "kasvi_rpt.rpt")
        escrever_csv(origem / "kasvi_virgula.csv", ",")
        escrever_csv(origem / "kasvi_ponto_virgula.csv", ";")
        escrever_rpt(origem / "outro_cliente.rpt")

        gerados = converter_pasta(origem, destino, clientes=("KASVI",))
        assert sorted(g.name for g in gerados) == ["kasvi_ponto_virgula.parquet", "kasvi_rpt.parquet",
                                                    "kasvi_virgula.parquet"]
        tabelas = [pq.read_table(g) for g in sorted(gerados)]
        assert all(t.equals(tabelas[0]) for t in tabelas), "rpt e csv divergem"

        df = tabelas[0].to_pandas()
        assert df["Consumo"].tolist() == [3.5, 0.0, 12.25]
        assert pd.isna(df["DiasADU"].iloc[1])
        assert str(df["Data"].dtype).startswith("datetime64")

        assert len(ler_parquet(destino, clientes=("kasvi",))) == 3 * (len(LINHAS) - 1)


def test_mesmo_nome_em_rpt_e_csv():
    with tempfile.TemporaryDirectory() as tmp:
        escrever_rpt(Path(tmp, "kasvi.rpt"))
        escrever_csv(Path(tmp, "kasvi.csv"))
        try:
            listar_fontes(tmp)
        except ValueError as erro:
            assert "kasvi" in str(erro)
        else:
            raise AssertionError("deveria recusar .rpt e .csv com o mesmo nome")


def test_csv_sem_cabecalho():
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp, "kasvi.csv")
        caminho.write_text("KASVI,2026-01-01,2,I01,1.0\n", encoding="utf-8")
        try:
            converter_pasta(Path(tmp), Path(tmp, "raw"))
        except ValueError as erro:
            assert "cabeçalho" in str(erro)
        else:
            raise AssertionError("deveria recusar CSV sem cabeçalho")
        assert not list(Path(tmp, "raw").rglob("*.parquet"))


def test_ordenacao():
    df = pd.DataFrame({
        "Cliente": ["K"] * 6,
        "IdBodega": ["2", "1", "2", "1", "2", "1"],
        "Referencia": ["A", "A", "A", "B", "B", "A"],
        "Data": pd.to_datetime(["2026-01-03", "2026-01-02", "2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"]),
    })
    for bagunca, mensagem in [(df, "Séries fora de ordem"),
                              (df.iloc[[2, 0, 5, 1, 3, 4]], "ordem alfabética"),
                              (df.iloc[[1, 5, 0, 2, 3, 4]], "Data fora de ordem")]:
        try:
            verificar_ordenacao(bagunca.reset_index(drop=True), CHAVE_SERIE)
        except ValueError as erro:
            assert mensagem in str(erro), erro
        else:
            raise AssertionError(mensagem)

    ordenado, _ = preparar_base(df, ())
    verificar_ordenacao(ordenado, CHAVE_SERIE)


if __name__ == "__main__":
    test_rpt_e_csv_geram_o_mesmo_parquet()
    test_mesmo_nome_em_rpt_e_csv()
    test_csv_sem_cabecalho()
    test_ordenacao()
    print("Conversão e ordenação: OK")
