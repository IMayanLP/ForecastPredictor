"""Conversão das exportações (.rpt ou .csv) para Parquet, incremental por mtime."""

import time
from pathlib import Path

import pyarrow.parquet as pq

from ddmrp_forecast.io.csv_reader import ler_csv_em_lotes
from ddmrp_forecast.io.rpt_reader import ler_rpt_em_lotes
from ddmrp_forecast.io.selecao import filtrar_por_cliente

LEITORES = {".rpt": ler_rpt_em_lotes, ".csv": ler_csv_em_lotes}


def listar_fontes(pasta: str | Path, clientes: tuple[str, ...] | None = None) -> list[Path]:
    arquivos = sorted(a for a in Path(pasta).rglob("*") if a.suffix.lower() in LEITORES)
    arquivos = filtrar_por_cliente(arquivos, clientes, pasta)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum {'/'.join(LEITORES)} em {pasta} para os clientes {clientes}")

    # .rpt e .csv com o mesmo nome gerariam o mesmo Parquet
    destinos = [a.with_suffix("") for a in arquivos]
    repetidos = {d.name for d in destinos if destinos.count(d) > 1}
    if repetidos:
        raise ValueError(f"Mesmo arquivo em .rpt e .csv: {', '.join(sorted(repetidos))}")
    return arquivos


def converter_para_parquet(origem: Path, destino: Path, tamanho_lote: int = 100_000) -> int:
    """Grava o Parquet em .tmp e só renomeia no fim. Retorna o número de linhas."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".tmp")
    escritor = None
    total = 0
    try:
        for tabela in LEITORES[origem.suffix.lower()](origem, tamanho_lote):
            escritor = escritor or pq.ParquetWriter(temporario, tabela.schema, compression="zstd")
            escritor.write_table(tabela)
            total += tabela.num_rows
    except Exception:
        if escritor:
            escritor.close()
        temporario.unlink(missing_ok=True)
        raise

    if escritor:
        escritor.close()
        temporario.replace(destino)  # só aparece o Parquet completo
    return total


def converter_pasta(pasta_origem: Path, pasta_parquet: Path, tamanho_lote: int = 100_000,
                    clientes: tuple[str, ...] | None = None) -> list[Path]:
    gerados = []
    for origem in listar_fontes(pasta_origem, clientes):
        destino = (pasta_parquet / origem.relative_to(pasta_origem)).with_suffix(".parquet")
        nome = origem.relative_to(pasta_origem)

        if destino.exists() and destino.stat().st_mtime >= origem.stat().st_mtime:
            print(f"{nome}: já convertido")
        else:
            inicio = time.time()
            print(f"{nome}: convertendo...")
            linhas = converter_para_parquet(origem, destino, tamanho_lote)
            print(f"{nome}: {linhas:,} linhas em {time.time() - inicio:,.0f}s")
        gerados.append(destino)
    return gerados
