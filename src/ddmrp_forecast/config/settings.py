from pathlib import Path

# src/ddmrp_forecast/config/settings.py → sobe 3 níveis até a raiz do projeto
RAIZ_PROJETO = Path(__file__).resolve().parents[3]

PASTA_SOURCE = RAIZ_PROJETO / "data" / "source"   # .rpt de entrada
PASTA_RAW = RAIZ_PROJETO / "data" / "raw"         # Parquet (próximo passo)

# Lista branca: só entram na base os arquivos cujo nome contém um destes clientes
# (sem diferenciar maiúsculas), ex.: hennings_consumos_01_a_03_2026.rpt
CLIENTES = ("kasvi")

# Linhas do .rpt processadas por vez (~0,5 GB de RAM por lote)
TAMANHO_LOTE = 100_000

# Escopo em que o buffer mestre "#X" anula o componente "X".
# ("Cliente",)            → dentro de cada cliente
# ("Cliente", "IdBodega") → só no mesmo centro de distribuição
# ()                      → global, igual ao Colab
ESCOPO_BUFFER_MESTRE = ("Cliente", "IdBodega")

# Série temporal = uma referência num centro de distribuição de um cliente
CHAVE_SERIE = ("Cliente", "Referencia", "IdBodega")  # também é a ordem do df final

# DiasADU = 0 na exportação vira este valor antes de qualquer etapa
DIAS_ADU_PADRAO = 30.0

# Pico: Consumo > Q3 + FATOR_IQR * IQR da série
FATOR_IQR = 3.0

# Q1/Q3 só dos dias com Consumo > 0 (False = igual ao Colab, com os zeros)
QUARTIS_SO_COM_CONSUMO = True

# Pico substituído pela média dos N registros anteriores (sem picos)
JANELA_LIMPEZA = 7

# Atributos da série copiados para os dias adicionados no calendário
COLUNAS_FIXAS = ["DiasADU", "ADI", "CV", "TiempoReposicion", "Descripcion", "Unidad"]

# Features sobre Consumo_Limpo (em dias, após completar o calendário)
LAGS = (1, 2, 3, 7, 14, 28, 365)
JANELAS_MEDIA = (7, 28, 90, 180, 365)

# SET DATEFIRST do SQL Server usado na exportação (7 = domingo é o dia 1, padrão us_english)
DATEFIRST_SQL = 7

# float32 corta pela metade a memória das features (~80M linhas)
TIPO_FEATURES = "float32"
