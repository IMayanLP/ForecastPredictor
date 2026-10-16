"""Conversão de .rpt do SSMS (largura fixa) para Parquet, em lotes."""

import codecs
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RODAPE = re.compile(r"^\s*\(\d+\s.*\)\s*$")  # (123 rows affected) / (123 linhas afetadas)
VALORES_NULOS = pa.array(["", "NULL"])

COLUNAS_DATA = ["Data"]
COLUNAS_INTEIRAS = ["DiaSemana", "DiaMes", "SemanaAno", "Mes", "TrimestreAno", "Ano"]
COLUNAS_DECIMAIS = [
    "Stock", "StockEnTransito", "StockTotal", "Consumo",
    "ADU", "DiasADU", "ADI", "CV", "TiempoReposicion",
]


def listar_rpts(pasta: str | Path) -> list[Path]:
    """Busca todos os .rpt da pasta e de suas subpastas, em ordem de nome."""
    arquivos = sorted(Path(pasta).rglob("*.rpt"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum .rpt encontrado em {pasta}")
    return arquivos


def detectar_encoding(caminho: Path) -> str:
    with open(caminho, "rb") as f:
        inicio = f.read(4)
    if inicio.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if inicio.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"  # padrão do SSMS
    return "utf-8"


def _tipar_coluna(nome: str, valores: pa.Array) -> pa.Array:
    valores = pc.utf8_trim_whitespace(valores)
    valores = pc.if_else(pc.is_in(valores, value_set=VALORES_NULOS), pa.scalar(None, pa.string()), valores)

    if nome in COLUNAS_DATA:
        # "2026-01-01 00:00:00.000" → só a data
        datas = pc.cast(pc.utf8_slice_codeunits(valores, 0, 10), pa.date32())
        return pc.cast(datas, pa.timestamp("s"))
    if nome in COLUNAS_INTEIRAS:
        return pc.cast(valores, pa.int16())
    if nome in COLUNAS_DECIMAIS:
        return pc.cast(valores, pa.float64())
    return valores


def _lote_para_tabela(linhas: list[str], largura: int, limites: list[tuple[int, int]],
                      colunas: list[str]) -> pa.Table:
    # todas as linhas com a mesma largura → matriz (linhas x caracteres)
    bloco = "".join(linha.ljust(largura)[:largura] for linha in linhas)
    matriz = np.frombuffer(bloco.encode("utf-32-le"), dtype=np.uint32).reshape(len(linhas), largura)

    arrays = []
    for nome, (a, b) in zip(colunas, limites):
        textos = np.ascontiguousarray(matriz[:, a:b]).view(f"<U{b - a}").ravel()
        arrays.append(_tipar_coluna(nome, pa.array(textos)))
    return pa.table(arrays, names=colunas)


def converter_rpt_para_parquet(origem: Path, destino: Path, tamanho_lote: int = 100_000) -> int:
    """Converte um .rpt em Parquet. Retorna o número de linhas gravadas."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".tmp")
    escritor = None
    total = 0

    try:
        with open(origem, encoding=detectar_encoding(origem)) as f:
            cabecalho = next(f).rstrip("\r\n")
            tracos = next(f).rstrip("\r\n")
            if not set(tracos.strip()) <= {"-", " "}:
                raise ValueError(f"{origem.name}: esperado .rpt em largura fixa (linha de traços)")

            inicios = [m.start() for m in re.finditer(r"-+", tracos)]
            largura = len(tracos)
            limites = list(zip(inicios, inicios[1:] + [largura]))
            colunas = [cabecalho[a:b].strip() for a, b in limites]

            lote = []
            for linha in f:
                linha = linha.rstrip("\r\n")
                if not linha.strip() or RODAPE.match(linha):
                    break  # fim dos dados
                lote.append(linha)
                if len(lote) >= tamanho_lote:
                    tabela = _lote_para_tabela(lote, largura, limites, colunas)
                    escritor = escritor or pq.ParquetWriter(temporario, tabela.schema, compression="zstd")
                    escritor.write_table(tabela)
                    total += len(lote)
                    lote = []

            if lote:
                tabela = _lote_para_tabela(lote, largura, limites, colunas)
                escritor = escritor or pq.ParquetWriter(temporario, tabela.schema, compression="zstd")
                escritor.write_table(tabela)
                total += len(lote)
    except Exception:
        if escritor:
            escritor.close()
        temporario.unlink(missing_ok=True)
        raise

    if escritor:
        escritor.close()
        temporario.replace(destino)  # só aparece o Parquet completo
    return total


def converter_pasta(pasta_rpt: Path, pasta_parquet: Path, tamanho_lote: int = 100_000) -> list[Path]:
    """Converte todos os .rpt, mantendo a estrutura de subpastas.

    Pula os arquivos cujo Parquet já existe e é mais novo que o .rpt.
    """
    gerados = []
    for origem in listar_rpts(pasta_rpt):
        destino = (pasta_parquet / origem.relative_to(pasta_rpt)).with_suffix(".parquet")
        nome = origem.relative_to(pasta_rpt)

        if destino.exists() and destino.stat().st_mtime >= origem.stat().st_mtime:
            print(f"{nome}: já convertido")
        else:
            inicio = time.time()
            print(f"{nome}: convertendo...")
            linhas = converter_rpt_para_parquet(origem, destino, tamanho_lote)
            print(f"{nome}: {linhas:,} linhas em {time.time() - inicio:,.0f}s")
        gerados.append(destino)
    return gerados


def listar_rpts(pasta: str | Path) -> list[Path]:
    """Busca todos os .rpt da pasta e de suas subpastas, em ordem de nome."""
    arquivos = sorted(Path(pasta).rglob("*.rpt"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum .rpt encontrado em {pasta}")
    return arquivos