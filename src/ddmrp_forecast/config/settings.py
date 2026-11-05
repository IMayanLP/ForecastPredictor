from pathlib import Path

# src/ddmrp_forecast/config/settings.py → sobe 3 níveis até a raiz do projeto
RAIZ_PROJETO = Path(__file__).resolve().parents[3]

PASTA_SOURCE = RAIZ_PROJETO / "data" / "source"   # .rpt de entrada
PASTA_RAW = RAIZ_PROJETO / "data" / "raw"         # Parquet (próximo passo)

# Lista branca: só entram na base os arquivos cujo nome contém um destes clientes
# (sem diferenciar maiúsculas), ex.: hennings_consumos_01_a_03_2026.rpt
CLIENTES = ("ceramfix")

# Leitura: só linhas com Data >= DATA_INICIO e só séries com ao menos VENDAS_MINIMAS dias com Consumo > 0
# nesse período (Ceramfix: 95% das séries nunca vendem). None = todo o histórico
DATA_INICIO = "2020-01-01"
VENDAS_MINIMAS = 1

# Linhas do .rpt processadas por vez (~0,5 GB de RAM por lote)
TAMANHO_LOTE = 100_000

# Escopo em que o buffer mestre "#X" anula o componente "X".
# ("Cliente",)            → dentro de cada cliente
# ("Cliente", "IdBodega") → só no mesmo centro de distribuição
# ()                      → global, igual ao Colab
ESCOPO_BUFFER_MESTRE = ("Cliente", "IdBodega")

# Perfis DDMRP sob encomenda (Descripcion terminada nestes sufixos) ficam fora do modelo: () = mantém todos
SUFIXOS_SOB_ENCOMENDA = ("TO",)

# Série temporal = uma referência num centro de distribuição de um cliente
CHAVE_SERIE = ("Cliente", "Referencia", "IdBodega")  # também é a ordem do df final

# DiasADU = 0 na exportação vira este valor antes de qualquer etapa
DIAS_ADU_PADRAO = 30.0

# Pico: Consumo > Q3 + FATOR_IQR * IQR da série
FATOR_IQR = 3.0

# Q1/Q3 só dos dias com Consumo > 0 (False = igual ao Colab, com os zeros)
QUARTIS_SO_COM_CONSUMO = True

# Valor que entra no lugar do pico no Consumo_Limpo:
# "limite" → LimiteSuperior da série | "media" → média dos JANELA_LIMPEZA registros anteriores sem pico (Colab)
SUBSTITUICAO_PICO = "limite"
JANELA_LIMPEZA = 7

# Atributos da série copiados para os dias adicionados no calendário
COLUNAS_FIXAS = ["DiasADU", "ADI", "CV", "TiempoReposicion", "Descripcion", "Unidad"]

# Lags e médias sobre o consumo sem picos (em dias, após completar o calendário): picos não entram nas features
COLUNA_FEATURES = "Consumo_Limpo"
LAGS = (1, 2, 3, 7, 14, 28, 365)
JANELAS_MEDIA = (7, 28, 90, 180, 365, 730)

# Intermitência (também sobre COLUNA_FEATURES, até t-1): Freq_N/TamanhoMedio_N, DiasDesdeVenda, TSB e SBA
JANELAS_FREQUENCIA = (28, 90, 365)
ALFA_INTERMITENCIA = 0.1

# SET DATEFIRST do SQL Server usado na exportação (7 = domingo é o dia 1, padrão us_english)
DATEFIRST_SQL = 7

# float32 corta pela metade a memória das features (~80M linhas)
TIPO_FEATURES = "float32"

# --- Treinamento (LightGBM) ---
PASTA_MODELO = RAIZ_PROJETO / "data" / "modelo"   # modelo, previsões do teste e métricas por série

# Alvo = média do Consumo_Limpo nos HORIZONTE_DIAS dias a partir de t (1 = consumo do próprio dia)
COLUNA_ALVO = "Consumo_Limpo"
HORIZONTE_DIAS = 28

# Split temporal pelos últimos dias da base: [treino][validação][teste]
# Cada base só tem origens cuja janela do alvo cabe inteira no próprio período
DIAS_TESTE = 90
DIAS_VALIDACAO = 60

# Só consumo sem picos; Lag_365/Media_365/Media_730 ficam NaN em séries com pouco histórico (LightGBM aceita)
# Partes da data fora: aprendiam sazonalidade falsa (Ceramfix, WAPE mediano 0,915 → 0,882 sem elas)
FEATURES = (
    "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14", "Lag_28", "Lag_365",
    "Media_7", "Media_28", "Media_90", "Media_180", "Media_365", "Media_730",
    "Freq_28", "Freq_90", "Freq_365", "TamanhoMedio_365", "DiasDesdeVenda", "DiasHistorico", "TSB", "SBA",
)

# Hurdle: features de quantidade (prefixos abaixo) e alvo divididos pela escala da série (1ª média > 0)
COLUNAS_ESCALA = ("Media_365", "Media_730")
PREFIXOS_ESCALAVEIS = ("Lag_", "Media_", "TamanhoMedio_", "TSB", "SBA")

# Concorrentes do modelo no teste (o ADU do banco fica fora: é um retrato único por série, calculado com o futuro)
BASELINES = ("ADU_recalculado", "Media_28")

PARAMETROS_ARVORES = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}
# Hurdle etapa 1: P(alguma venda em [t, t+H-1]) | etapa 2: log(alvo / escala) nas linhas com venda (mediana)
PARAMETROS_CLASSIFICADOR = PARAMETROS_ARVORES | {"objective": "binary"}
PARAMETROS_REGRESSOR = PARAMETROS_ARVORES | {"objective": "regression_l1", "metric": "l1"}
# Modelo de uma etapa (treinar_lgbm), mantido para comparação
PARAMETROS_LGBM = PARAMETROS_ARVORES | {"objective": "tweedie", "tweedie_variance_power": 1.3}
RODADAS_MAXIMAS = 2_000
PARADA_ANTECIPADA = 100

# --- Busca de hiperparâmetros (etapa do treino: roda a cada execução do main) ---
# Base pronta (features + Alvo + Base) gravada pelo main, um arquivo por conjunto de CLIENTES
PASTA_FEATURES = RAIZ_PROJETO / "data" / "features"
REAPROVEITAR_BASE = False   # True: lê a base salva dos mesmos CLIENTES e pula as etapas 1–8
# Registro por base: <clientes>_<id>/tentativas.jsonl; a mesma base retoma de onde parou, base nova abre outro
PASTA_BUSCA = PASTA_MODELO / "busca"
BUSCAR_HIPERPARAMETROS = True   # False: treina direto com PARAMETROS_CLASSIFICADOR / PARAMETROS_REGRESSOR
# Tentativa 0 de cada etapa = parâmetros acima (referência); as demais sorteiam ESPACO_BUSCA sobre eles
TENTATIVAS_CLASSIFICADOR = 40   # escolha por log loss na validação
TENTATIVAS_REGRESSOR = 60       # escolha pelo WAPE do tamanho na validação (linhas com consumo)
SEMENTE_BUSCA = 2026
# Os MELHORES_POR_ETAPA de cada etapa são combinados e avaliados pelo WAPE mediano por série; as melhores
# combinações são repetidas com seed + SEMENTES_EXTRAS e vence a média. Combinação cujo WAPE global piora mais
# que TOLERANCIA_WAPE_GLOBAL contra os parâmetros acima é descartada
MELHORES_POR_ETAPA = 3
SEMENTES_EXTRAS = (1, 2)
TOLERANCIA_WAPE_GLOBAL = 0.02
# (tipo, limites): real/log_real/log_int entre mínimo e máximo; zero_ou_log = 0 em 1/3 das vezes; escolha = lista
ESPACO_BUSCA = {
    "learning_rate": ("log_real", 0.02, 0.1),
    "num_leaves": ("log_int", 15, 255),
    "max_depth": ("escolha", (-1, 4, 6, 8, 10, 12)),
    "min_data_in_leaf": ("log_int", 20, 2_000),
    "feature_fraction": ("real", 0.5, 1.0),
    "bagging_fraction": ("real", 0.5, 1.0),
    "lambda_l1": ("zero_ou_log", 1e-3, 10.0),
    "lambda_l2": ("zero_ou_log", 1e-3, 10.0),
    "min_gain_to_split": ("real", 0.0, 1.0),
}
ESPACO_BUSCA_REGRESSOR = ESPACO_BUSCA | {"objective": ("escolha", ("regression_l1", "huber", "regression"))}
