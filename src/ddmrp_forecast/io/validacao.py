"""Checagem da base de consumos antes do pipeline."""

import pandas as pd

COLUNAS_OBRIGATORIAS = [
    "Data", "Cliente", "IdBodega", "Referencia", "Consumo",
    "DiasADU", "ADI", "CV", "TiempoReposicion", "Descripcion", "Unidad",
]


def validar_consumos(df: pd.DataFrame, chave: tuple[str, ...]) -> list[str]:
    """Lista os problemas encontrados. Vazia = base pronta para o pipeline."""
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        return [f"Colunas ausentes na exportação: {', '.join(faltando)}"]

    problemas = []
    datas_nulas = int(df["Data"].isna().sum())
    if datas_nulas:
        problemas.append(f"{datas_nulas:,} registros sem Data")

    consumo_nulo = int(df["Consumo"].isna().sum())
    if consumo_nulo:
        problemas.append(f"{consumo_nulo:,} registros com Consumo nulo (viram 0 no calendário)")

    consumo_negativo = int((df["Consumo"] < 0).sum())
    if consumo_negativo:
        problemas.append(f"{consumo_negativo:,} registros com Consumo negativo")

    duplicados = int(df.duplicated([*chave, "Data"]).sum())
    if duplicados:
        problemas.append(f"{duplicados:,} registros duplicados por série e Data")

    sem_dias_adu = df.groupby(list(chave), observed=True)["DiasADU"].count().eq(0).sum()
    if sem_dias_adu:
        problemas.append(f"{sem_dias_adu:,} séries sem DiasADU (ADU_recalculado fica NaN)")
    return problemas


def resumir_periodo(df: pd.DataFrame) -> str:
    return (f"{df['Data'].min():%Y-%m-%d} a {df['Data'].max():%Y-%m-%d}, "
            f"{df['Data'].nunique():,} dias distintos")
