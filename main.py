import time

from ddmrp_forecast.config.settings import (
    ALFA_INTERMITENCIA, BASELINES, CHAVE_SERIE, CLIENTES, COLUNA_ALVO, COLUNA_FEATURES, COLUNAS_ESCALA, COLUNAS_FIXAS,
    DATA_INICIO, DATEFIRST_SQL, DIAS_ADU_PADRAO, DIAS_TESTE, DIAS_VALIDACAO, ESCOPO_BUFFER_MESTRE, FATOR_IQR, FEATURES,
    HORIZONTE_DIAS, JANELA_LIMPEZA, JANELAS_FREQUENCIA, JANELAS_MEDIA, LAGS, PARADA_ANTECIPADA,
    PARAMETROS_CLASSIFICADOR, PARAMETROS_REGRESSOR, PASTA_MODELO, PASTA_RAW, PASTA_SOURCE, PREFIXOS_ESCALAVEIS,
    QUARTIS_SO_COM_CONSUMO, RODADAS_MAXIMAS, SUBSTITUICAO_PICO, SUFIXOS_SOB_ENCOMENDA, TAMANHO_LOTE, TIPO_FEATURES,
    VENDAS_MINIMAS,
)
from ddmrp_forecast.features.datas import adicionar_partes_da_data
from ddmrp_forecast.features.defasagens import criar_lags, criar_medias_moveis
from ddmrp_forecast.features.intermitencia import criar_frequencias, criar_recencia, criar_tsb_sba
from ddmrp_forecast.io.parquet_store import ler_series_com_venda
from ddmrp_forecast.io.conversao import converter_pasta
from ddmrp_forecast.io.validacao import resumir_periodo, validar_consumos
from ddmrp_forecast.model.alvo import criar_alvo
from ddmrp_forecast.model.metricas import comparar_modelos, metricas_por_serie
from ddmrp_forecast.model.split import datas_corte, resumir_bases, separar_bases
from ddmrp_forecast.model.hurdle import calcular_escala, prever_hurdle, treinar_hurdle
from ddmrp_forecast.model.treino import importancia
from ddmrp_forecast.preprocess.adu import corrigir_dias_adu, recalcular_adu
from ddmrp_forecast.preprocess.auditoria import auditar_adu, auditar_features, auditar_picos
from ddmrp_forecast.preprocess.calendario import completar_calendario
from ddmrp_forecast.preprocess.picos import limpar_picos, marcar_picos, resumir_picos
from ddmrp_forecast.preprocess.preparacao import (
    preparar_base, remover_sob_encomenda, verificar_ordenacao, zerar_consumos_negativos,
)


def etapa(nome: str, inicio: float) -> float:
    print(f"[{nome}] {time.time() - inicio:,.0f}s")
    return time.time()


def main() -> None:
    # 1) .rpt/.csv → Parquet (só converte o que for novo ou alterado)
    converter_pasta(PASTA_SOURCE, PASTA_RAW, TAMANHO_LOTE, CLIENTES)

    # 2) leitura e checagem do Parquet
    inicio = time.time()
    df, resumo_leitura = ler_series_com_venda(PASTA_RAW, CLIENTES, CHAVE_SERIE, DATA_INICIO, VENDAS_MINIMAS)
    print(f"Leitura a partir de {DATA_INICIO or 'todo o histórico'}:")
    for descricao, quantidade in resumo_leitura.items():
        print(f"  · {descricao}: {quantidade:,}")
    print(f"Registros lidos: {len(df):,} ({df.memory_usage(deep=True).sum() / 1e9:,.2f} GB em memória)")
    print(f"Período: {resumir_periodo(df)}")
    problemas = validar_consumos(df, CHAVE_SERIE)
    for problema in problemas:
        print(f"  ⚠ {problema}")
    if any(p.startswith("Colunas ausentes") for p in problemas):
        raise SystemExit("Base incompleta: corrigir a exportação do SSMS antes de seguir.")
    df, negativos = zerar_consumos_negativos(df)
    print(f"Consumo negativo → 0: {negativos:,} registros")
    df, corrigidos = corrigir_dias_adu(df, DIAS_ADU_PADRAO)
    print(f"DiasADU 0 → {DIAS_ADU_PADRAO:g}: {corrigidos:,} registros")
    # fronteiras do split antes dos picos: os quartis só podem ver o período de treino
    inicio_validacao, inicio_teste = datas_corte(df["Data"].max(), DIAS_TESTE, DIAS_VALIDACAO)
    print(f"Validação a partir de {inicio_validacao:%Y-%m-%d}, teste a partir de {inicio_teste:%Y-%m-%d}")
    inicio = etapa("leitura", inicio)

    # 3) buffers mestres + ordenação
    df, removidas = preparar_base(df, ESCOPO_BUFFER_MESTRE)
    print(f"Linhas de componentes removidas: {removidas:,}")
    print(f"Registros após remoção dos componentes: {len(df):,}")
    df, linhas_encomenda, series_encomenda = remover_sob_encomenda(df, CHAVE_SERIE, SUFIXOS_SOB_ENCOMENDA)
    print(f"Sob encomenda (perfil terminado em {SUFIXOS_SOB_ENCOMENDA}) fora do modelo: "
          f"{series_encomenda:,} séries, {linhas_encomenda:,} registros; restam {len(df):,}")
    verificar_ordenacao(df, CHAVE_SERIE)
    inicio = etapa("buffers mestres", inicio)

    # 4) picos
    df, _ = marcar_picos(df, CHAVE_SERIE, FATOR_IQR, QUARTIS_SO_COM_CONSUMO, ate=inicio_validacao)
    print(df["EhPico"].value_counts().to_string())
    print(resumir_picos(df, CHAVE_SERIE).head(20).to_string())
    df = limpar_picos(df, CHAVE_SERIE, JANELA_LIMPEZA, SUBSTITUICAO_PICO)
    inicio = etapa("picos", inicio)

    # 5) calendário diário completo
    registros = len(df)
    df, dias_novos = completar_calendario(df, CHAVE_SERIE, COLUNAS_FIXAS)
    print(f"Registros originais: {registros:,}")
    print(f"Registros após completar: {len(df):,}")
    print(f"Dias adicionados: {dias_novos:,}")
    inicio = etapa("calendário", inicio)

    # 6) ADU recalculado
    print(df["DiasADU"].value_counts().head(20).to_string())
    df = recalcular_adu(df, CHAVE_SERIE)
    inicio = etapa("ADU", inicio)

    # 7) features de treino
    df = criar_lags(df, CHAVE_SERIE, LAGS, coluna=COLUNA_FEATURES, tipo=TIPO_FEATURES)
    df = criar_medias_moveis(df, CHAVE_SERIE, JANELAS_MEDIA, coluna=COLUNA_FEATURES, tipo=TIPO_FEATURES)
    df = criar_frequencias(df, CHAVE_SERIE, JANELAS_FREQUENCIA, coluna=COLUNA_FEATURES, tipo=TIPO_FEATURES)
    df = criar_recencia(df, CHAVE_SERIE, coluna=COLUNA_FEATURES, tipo=TIPO_FEATURES)
    df = criar_tsb_sba(df, CHAVE_SERIE, ALFA_INTERMITENCIA, coluna=COLUNA_FEATURES, tipo=TIPO_FEATURES)
    df, divergencias = adicionar_partes_da_data(df, DATEFIRST_SQL)
    for coluna, quantidade in divergencias.items():
        if quantidade:
            print(f"  ⚠ {coluna}: {quantidade:,} valores exportados diferem do calculado (conferir DATEFIRST_SQL)")
    etapa("features", inicio)

    # 8) auditoria da base final
    verificar_ordenacao(df, CHAVE_SERIE)
    print(f"Ordenação por {[*CHAVE_SERIE, 'Data']}: ok")
    total_erros = 0
    for nome, (informacoes, erros) in [("Picos", auditar_picos(df, JANELA_LIMPEZA, SUBSTITUICAO_PICO)),
                                       ("ADU", auditar_adu(df, CHAVE_SERIE)),
                                       ("Features", auditar_features(df, CHAVE_SERIE, LAGS, JANELAS_MEDIA,
                                                                     COLUNA_FEATURES))]:
        print(f"{nome}:")
        for descricao, quantidade in informacoes.items():
            print(f"  · {descricao}: {quantidade:,}")
        for descricao, quantidade in erros.items():
            print(f"  {'✗' if quantidade else '✓'} {descricao}: {quantidade:,}")
            total_erros += quantidade
    if total_erros:
        raise SystemExit(f"Auditoria encontrou {total_erros:,} inconsistências")

    print(f"Base de treino: {len(df):,} linhas x {df.shape[1]} colunas "
          f"({df.memory_usage(deep=True).sum() / 1e9:,.2f} GB)")
    print(df.head(10).to_string())

    # 9) treinamento: alvo à frente, split temporal, hurdle (P(consumo) × tamanho) e comparação no teste
    inicio = time.time()
    df = criar_alvo(df, CHAVE_SERIE, COLUNA_ALVO, HORIZONTE_DIAS)
    base = separar_bases(df, HORIZONTE_DIAS, inicio_validacao, inicio_teste)
    print(f"Alvo: média de {COLUNA_ALVO} nos próximos {HORIZONTE_DIAS} dias")
    print(resumir_bases(df, base).to_string())

    escala = calcular_escala(df, COLUNAS_ESCALA)
    print(f"Linhas sem escala ({'/'.join(COLUNAS_ESCALA)} = 0, previsão 0 e fora do treino): {(escala == 0).sum():,}")
    classificador, regressor = treinar_hurdle(df, base, FEATURES, PREFIXOS_ESCALAVEIS, escala, PARAMETROS_CLASSIFICADOR,
                                              PARAMETROS_REGRESSOR, RODADAS_MAXIMAS, PARADA_ANTECIPADA)
    for nome, modelo in [("Etapa 1 — classificador (P(consumo))", classificador),
                         ("Etapa 2 — regressor (tamanho)", regressor)]:
        print(f"{nome}: melhor rodada {modelo.best_iteration:,}")
        print(importancia(modelo).to_string())

    linhas_teste = (base == "teste").fillna(False).to_numpy()
    teste = df.loc[linhas_teste,
                   [*CHAVE_SERIE, "Data", "Consumo", COLUNA_ALVO, "EhPico", "Descripcion", "Alvo", *BASELINES, *FEATURES]]
    teste = teste.loc[:, ~teste.columns.duplicated()].copy()
    teste["Previsao"], teste["Probabilidade"] = prever_hurdle(classificador, regressor, df, FEATURES,
                                                             PREFIXOS_ESCALAVEIS, escala, linhas_teste)
    inicio = etapa("treinamento", inicio)

    concorrentes = ("Previsao", *BASELINES)
    comuns = teste[list(BASELINES)].notna().all(axis=1)
    por_serie = metricas_por_serie(teste[comuns], CHAVE_SERIE, concorrentes)
    print(f"Teste — linhas com todos os baselines ({comuns.sum():,} de {len(teste):,}):")
    print(comparar_modelos(por_serie, CHAVE_SERIE, "Previsao").to_string(float_format="{:,.4f}".format))
    print("Teste — modelo em todas as linhas:")
    print(comparar_modelos(metricas_por_serie(teste, CHAVE_SERIE, ("Previsao",)), CHAVE_SERIE, "Previsao")
          .to_string(float_format="{:,.4f}".format))

    PASTA_MODELO.mkdir(parents=True, exist_ok=True)
    classificador.save_model(PASTA_MODELO / "hurdle_classificador.txt", num_iteration=classificador.best_iteration)
    regressor.save_model(PASTA_MODELO / "hurdle_regressor.txt", num_iteration=regressor.best_iteration)
    teste.to_parquet(PASTA_MODELO / "teste_previsoes.parquet", index=False)
    por_serie.to_parquet(PASTA_MODELO / "metricas_por_serie.parquet", index=False)
    print(f"Modelo, previsões do teste e métricas por série em {PASTA_MODELO}")
    etapa("métricas", inicio)


if __name__ == "__main__":
    main()
