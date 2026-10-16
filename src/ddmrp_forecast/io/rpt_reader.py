"""Leitura de .rpt do SSMS (largura fixa), em lotes."""

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa

from ddmrp_forecast.io.tipagem import detectar_encoding, tipar_coluna

RODAPE = re.compile(r"^\s*\(\d+\s.*\)\s*$")  # (123 rows affected) / (123 linhas afetadas)


def _lote_para_tabela(linhas: list[str], largura: int, limites: list[tuple[int, int]],
                      colunas: list[str]) -> pa.Table:
    # todas as linhas com a mesma largura → matriz (linhas x caracteres)
    bloco = "".join(linha.ljust(largura)[:largura] for linha in linhas)
    matriz = np.frombuffer(bloco.encode("utf-32-le"), dtype=np.uint32).reshape(len(linhas), largura)

    arrays = []
    for nome, (a, b) in zip(colunas, limites):
        textos = np.ascontiguousarray(matriz[:, a:b]).view(f"<U{b - a}").ravel()
        arrays.append(tipar_coluna(nome, pa.array(textos)))
    return pa.table(arrays, names=colunas)


def ler_rpt_em_lotes(origem: Path, tamanho_lote: int = 100_000) -> Iterator[pa.Table]:
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
                yield _lote_para_tabela(lote, largura, limites, colunas)
                lote = []

        if lote:
            yield _lote_para_tabela(lote, largura, limites, colunas)
