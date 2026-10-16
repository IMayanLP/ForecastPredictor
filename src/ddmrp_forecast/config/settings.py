from pathlib import Path

# src/ddmrp_forecast/config/settings.py → sobe 3 níveis até a raiz do projeto
RAIZ_PROJETO = Path(__file__).resolve().parents[3]

PASTA_SOURCE = RAIZ_PROJETO / "data" / "source"   # .rpt de entrada
PASTA_RAW = RAIZ_PROJETO / "data" / "raw"         # Parquet (próximo passo)

# Linhas do .rpt processadas por vez (~0,5 GB de RAM por lote)
TAMANHO_LOTE = 100_000

# Escopo em que o buffer mestre "#X" anula o componente "X".
# ("Cliente",)            → dentro de cada cliente
# ("Cliente", "IdBodega") → só no mesmo centro de distribuição
# ()                      → global, igual ao Colab
ESCOPO_BUFFER_MESTRE = ("Cliente", "IdBodega")