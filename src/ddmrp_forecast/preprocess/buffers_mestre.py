"""Buffers mestres: "#X" já agrega a demanda do componente "X".

Mantém o mestre e remove o componente para não duplicar demanda.
"""

import pandas as pd

PREFIXO_MESTRE = "#"

def remover_componentes(df: pd.DataFrame, escopo: tuple[str, ...] = ("Cliente",)) -> tuple[pd.DataFrame, int]:
    chaves = [*escopo, "Referencia"]
    eh_mestre = df["Referencia"].str.startswith(PREFIXO_MESTRE, na=False)

    # para cada mestre "#X", o componente a remover é "X" no mesmo escopo
    componentes = df.loc[eh_mestre, chaves].drop_duplicates()
    componentes["Referencia"] = componentes["Referencia"].str[len(PREFIXO_MESTRE):]

    eh_componente = (
        pd.MultiIndex.from_frame(df[chaves]).isin(pd.MultiIndex.from_frame(componentes))
        & ~eh_mestre.to_numpy()
    )
    return df.loc[~eh_componente].copy(), int(eh_componente.sum())