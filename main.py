import json
import time

import pandas as pd

from ddmrp_forecast.config.settings import (
    ALFA_INTERMITENCIA, AMOSTRA_SERIES, BASELINES, BUSCAR_HIPERPARAMETROS, CHAVE_SERIE, CLIENTES, COLUNA_ALVO,
    COLUNA_FEATURES,
    COLUNAS_ESCALA, COLUNAS_FIXAS, CORTE_TIPO_PREVISAO, CORTES_CLASSES, DATA_INICIO, DATEFIRST_SQL, DIAS_ADU_PADRAO,
    DIAS_TESTE, DIAS_VALIDACAO, DOBRAS_CONFIABILIDADE, ESCOPO_BUFFER_MESTRE, ESPACO_BUSCA, ESPACO_BUSCA_REGRESSOR,
    FAIXAS_REGRAS, FATOR_IQR, FEATURES, FEATURES_CONFIABILIDADE, FEATURES_SERIE_CONFIABILIDADE, HORIZONTE_DIAS,
    JANELA_INTERVALOS, JANELA_LIMPEZA, JANELA_PICOS, JANELA_TAMANHO, JANELAS_FREQUENCIA, JANELAS_MEDIA, LAGS,
    LINHAS_MINIMAS_CLASSE, MELHORES_POR_ETAPA, PARADA_ANTECIPADA, PARADA_CONFIABILIDADE, PARAMETROS_CLASSIFICADOR,
    PARAMETROS_CONFIABILIDADE, PARAMETROS_REGRESSOR, PASTA_BUSCA, PASTA_FEATURES, PASTA_MODELO, PASTA_RAW, PASTA_SOURCE,
    PREFIXOS_ESCALAVEIS, QUARTIS_SO_COM_CONSUMO, REAPROVEITAR_BASE, REGRAS_CONFIABILIDADE, REPETICOES_BOOTSTRAP,
    RODADAS_CONFIABILIDADE, RODADAS_MAXIMAS, ROTULOS_CLASSES, SEMENTE_BUSCA, SEMENTES_EXTRAS, SUBSTITUICAO_PICO,
    SUFIXOS_SOB_ENCOMENDA, TAMANHO_LOTE, TENTATIVAS_CLASSIFICADOR, TENTATIVAS_REGRESSOR, TIPO_FEATURES,
    TOLERANCIA_ACERTO, TOLERANCIA_WAPE_GLOBAL, VENDAS_MINIMAS,
)
from ddmrp_forecast.features.datas import adicionar_partes_da_data
from ddmrp_forecast.features.defasagens import criar_lags, criar_medias_moveis
from ddmrp_forecast.features.intermitencia import (
    criar_features_confiabilidade, criar_frequencias, criar_recencia, criar_tsb_sba,
)
from ddmrp_forecast.io.parquet_store import ler_series_com_venda
from ddmrp_forecast.io.conversao import converter_pasta
from ddmrp_forecast.io.validacao import resumir_periodo, validar_consumos
from ddmrp_forecast.model.alvo import criar_alvo
from ddmrp_forecast.model.confiabilidade import (
    ajustar_confiabilidade, aplicar_confiabilidade, bootstrap_por_serie, classificar, comparar_com_regras, eh_monotona,
    gravar_confiabilidade, marcar_acerto, preparar_sinais, relatorio_por_recorte, tipo_previsao,
)
from ddmrp_forecast.model.metricas import comparar_modelos, comparar_por_segmento, metricas_por_serie, segmentar_abc
from ddmrp_forecast.model.split import (
    datas_corte_por_cliente, inicio_da_base, por_linha, resumir_bases, separar_bases,
)
from ddmrp_forecast.model.busca import executar_busca, identificar_base
from ddmrp_forecast.model.hurdle import (
    calcular_escala, montar_matrizes, prever_hurdle, treinar_classificador, treinar_regressor,
)
from ddmrp_forecast.model.treino import importancia
from ddmrp_forecast.preprocess.adu import corrigir_dias_adu, recalcular_adu
from ddmrp_forecast.preprocess.auditoria import auditar_adu, auditar_features, auditar_picos
from ddmrp_forecast.preprocess.calendario import completar_calendario
from ddmrp_forecast.preprocess.picos import limpar_picos, marcar_picos, resumir_picos
from ddmrp_forecast.preprocess.preparacao import (
    preparar_base, remover_duplicados_identicos, remover_sob_encomenda, verificar_ordenacao,
    zerar_consumos_negativos,
)


def etapa(nome: str, inicio: float) -> float:
    print(f"[{nome}] {time.time() - inicio:,.0f}s")
    return time.time()


def nome_base(clientes: str | tuple[str, ...]) -> str:
    return "_".join(sorted((clientes,) if isinstance(clientes, str) else clientes)).lower()


def montar_base() -> pd.DataFrame:
    # 1) .rpt/.csv → Parquet (só converte o que for novo ou alterado)
    converter_pasta(PASTA_SOURCE, PASTA_RAW, TAMANHO_LOTE, CLIENTES)

    # 2) leitura e checagem do Parquet
    inicio = time.time()
    df, resumo_leitura = ler_series_com_venda(PASTA_RAW, CLIENTES, CHAVE_SERIE, DATA_INICIO, VENDAS_MINIMAS,
                                              amostra=AMOSTRA_SERIES)
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
    df, duplicados = remover_duplicados_identicos(df, CHAVE_SERIE)
    print(f"Linhas repetidas idênticas removidas: {duplicados:,}")
    df, negativos = zerar_consumos_negativos(df)
    print(f"Consumo negativo → 0: {negativos:,} registros")
    df, corrigidos = corrigir_dias_adu(df, DIAS_ADU_PADRAO)
    print(f"DiasADU 0 → {DIAS_ADU_PADRAO:g}: {corrigidos:,} registros")
    # fronteiras do split antes dos picos: os quartis só podem ver o período de treino
    # cada cliente tem o seu corte: teste nos últimos DIAS_TESTE dias da própria exportação
    cortes = datas_corte_por_cliente(df, DIAS_TESTE, DIAS_VALIDACAO)
    for cliente, corte in cortes.iterrows():
        print(f"{cliente}: validação a partir de {corte['InicioValidacao']:%Y-%m-%d}, "
              f"teste a partir de {corte['InicioTeste']:%Y-%m-%d}, dados até {corte['Fim']:%Y-%m-%d}")
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
    df, _ = marcar_picos(df, CHAVE_SERIE, FATOR_IQR, QUARTIS_SO_COM_CONSUMO,
                         ate=por_linha(df, cortes["InicioValidacao"]))
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

    # 9) alvo à frente e split temporal; a base fica só com o que o treino e as métricas usam
    df = criar_alvo(df, CHAVE_SERIE, COLUNA_ALVO, HORIZONTE_DIAS)
    df["Base"] = separar_bases(df, HORIZONTE_DIAS, por_linha(df, cortes["InicioValidacao"]),
                               por_linha(df, cortes["InicioTeste"]))
    colunas = [*CHAVE_SERIE, "Data", "Consumo", COLUNA_ALVO, "EhPico", "Descripcion", "Alvo", "Base", *BASELINES,
               *FEATURES, *COLUNAS_ESCALA]
    return df[list(dict.fromkeys(colunas))]


def com_segmento(linhas: pd.DataFrame, segmentos: pd.Series) -> pd.DataFrame:
    # série fora da segmentação conta como C
    linhas = linhas.join(segmentos, on=list(CHAVE_SERIE))
    linhas["Segmento"] = linhas["Segmento"].fillna("C")
    return linhas


def classificar_confiabilidade(df: pd.DataFrame, base: pd.Series, classificador, regressor, escala,
                               teste: pd.DataFrame, segmentos: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # 11) confiabilidade: ajuste só na validação, aplicada e medida no teste
    linhas_validacao = (base == "validacao").fillna(False).to_numpy()
    # Probabilidade e PrevSobreEscala saem da previsão, logo abaixo
    colunas = [*CHAVE_SERIE, "Data", "Alvo", *(c for c in FEATURES_CONFIABILIDADE if c in df.columns)]
    validacao = df.loc[linhas_validacao, colunas].copy()
    previsao, probabilidade = prever_hurdle(classificador, regressor, df, FEATURES, PREFIXOS_ESCALAVEIS, escala,
                                            linhas_validacao)
    preparar_sinais(validacao, previsao, probabilidade, escala[linhas_validacao])
    ajuste = ajustar_confiabilidade(validacao, CHAVE_SERIE, FEATURES_CONFIABILIDADE, TOLERANCIA_ACERTO,
                                    PARAMETROS_CONFIABILIDADE, DOBRAS_CONFIABILIDADE, RODADAS_CONFIABILIDADE,
                                    PARADA_CONFIABILIDADE)
    print(f"Confiabilidade — ajuste na validação ({len(validacao):,} linhas): AUC fora da dobra "
          f"{ajuste['auc_fora_da_dobra']:.3f}, rodadas {ajuste['rodadas_por_dobra']}, "
          f"taxa de acerto sem escala {ajuste['taxa_sem_escala']:.1%}")

    linhas_teste = (base == "teste").fillna(False).to_numpy()
    preparar_sinais(teste, teste["Previsao"].to_numpy(float), teste["Probabilidade"].to_numpy(float), escala[linhas_teste])
    teste["Confiabilidade"] = aplicar_confiabilidade(teste, ajuste)
    teste["ClasseConfiabilidade"] = classificar(teste["Confiabilidade"].to_numpy(), CORTES_CLASSES, ROTULOS_CLASSES)
    teste["TipoPrevisao"] = tipo_previsao(teste["Probabilidade"].to_numpy(), CORTE_TIPO_PREVISAO)
    teste["Acerto"] = marcar_acerto(teste["Alvo"].to_numpy(float), teste["Previsao"].to_numpy(float),
                                    teste["Escala"].to_numpy(float), TOLERANCIA_ACERTO)
    teste = com_segmento(teste, segmentos)

    formato = "{:,.3f}".format
    relatorio = relatorio_por_recorte(teste, CHAVE_SERIE)
    for recorte, tabela in relatorio.groupby(level="Recorte", sort=False):
        aviso = "" if eh_monotona(tabela, LINHAS_MINIMAS_CLASSE) else "  ⚠ acerto não monótono"
        print(f"Teste — confiabilidade, {recorte}:{aviso}")
        print(tabela.droplevel("Recorte").to_string(float_format=formato))

    com_escala = teste[teste["Escala"] > 0]
    intervalo, monotonos = bootstrap_por_serie(com_escala, CHAVE_SERIE, REPETICOES_BOOTSTRAP, SEMENTE_BUSCA,
                                               LINHAS_MINIMAS_CLASSE)
    print(f"Teste — bootstrap por série ({REPETICOES_BOOTSTRAP} sorteios): acerto monótono em {monotonos:.1%}")
    print(intervalo.to_string(float_format=formato))

    # regras simples nas mesmas linhas; segmentos da validação pelo volume antes dela
    antes_validacao = (df["Data"] < inicio_da_base(df, linhas_validacao)).to_numpy()
    segmentos_validacao = segmentar_abc(df, CHAVE_SERIE, COLUNA_ALVO, antes_validacao)
    validacao = com_segmento(validacao, segmentos_validacao)
    validacao["Acerto"] = marcar_acerto(validacao["Alvo"].to_numpy(float), validacao["Previsao"].to_numpy(float),
                                        validacao["Escala"].to_numpy(float), TOLERANCIA_ACERTO)
    regras = comparar_com_regras(validacao[validacao["Escala"] > 0], com_escala, REGRAS_CONFIABILIDADE, FAIXAS_REGRAS,
                                 CORTES_CLASSES, ROTULOS_CLASSES)
    print("Teste — modelo × regras simples (AUC, Brier e acerto real em cada classe):")
    print(regras.to_string(float_format=formato))
    return teste, relatorio, ajuste


def treinar(df: pd.DataFrame) -> None:
    # 10) busca de hiperparâmetros na validação, treino final e comparação no teste (uma vez, com o vencedor)
    inicio = time.time()
    base = df["Base"]
    print(f"Alvo: média de {COLUNA_ALVO} nos próximos {HORIZONTE_DIAS} dias")
    print(resumir_bases(df, base).to_string())
    clientes = df["Cliente"].unique()
    if len(clientes) > 1:
        for cliente in clientes:
            linhas = (df["Cliente"] == cliente).to_numpy()
            print(f"Bases — {cliente}:")
            print(resumir_bases(df.loc[linhas, ["Data", "Alvo"]], base[linhas]).to_string())

    escala = calcular_escala(df, COLUNAS_ESCALA)
    print(f"Linhas sem escala ({'/'.join(COLUNAS_ESCALA)} = 0, previsão 0 e fora do treino): {(escala == 0).sum():,}")
    matrizes = montar_matrizes(df, base, FEATURES, PREFIXOS_ESCALAVEIS, escala)
    if BUSCAR_HIPERPARAMETROS:
        configuracao = {"features": FEATURES, "prefixos": PREFIXOS_ESCALAVEIS, "escala": COLUNAS_ESCALA,
                        "classificador": PARAMETROS_CLASSIFICADOR, "regressor": PARAMETROS_REGRESSOR,
                        "espaco": ESPACO_BUSCA, "espaco_regressor": ESPACO_BUSCA_REGRESSOR, "semente": SEMENTE_BUSCA,
                        "rodadas": RODADAS_MAXIMAS, "parada": PARADA_ANTECIPADA, "melhores": MELHORES_POR_ETAPA,
                        "sementes_extras": SEMENTES_EXTRAS, "tolerancia": TOLERANCIA_WAPE_GLOBAL}
        pasta = PASTA_BUSCA / f"{nome_base(CLIENTES)}_{identificar_base(df, base, configuracao)}"
        busca = executar_busca(df, base, matrizes, CHAVE_SERIE, COLUNA_ALVO, escala, pasta, PARAMETROS_CLASSIFICADOR,
                               PARAMETROS_REGRESSOR, ESPACO_BUSCA, ESPACO_BUSCA_REGRESSOR, TENTATIVAS_CLASSIFICADOR,
                               TENTATIVAS_REGRESSOR, SEMENTE_BUSCA, MELHORES_POR_ETAPA, SEMENTES_EXTRAS,
                               TOLERANCIA_WAPE_GLOBAL, RODADAS_MAXIMAS, PARADA_ANTECIPADA)
        modelos = busca.pop("modelos")
        print(f"Vencedor: classificador {busca['classificador']}, regressor {busca['regressor']} (registro em {pasta})")
        inicio = etapa("busca", inicio)
    else:
        busca = {"parametros_classificador": PARAMETROS_CLASSIFICADOR, "parametros_regressor": PARAMETROS_REGRESSOR}
        modelos = {"vencedor": (treinar_classificador(matrizes, PARAMETROS_CLASSIFICADOR, RODADAS_MAXIMAS, PARADA_ANTECIPADA),
                                treinar_regressor(matrizes, PARAMETROS_REGRESSOR, RODADAS_MAXIMAS, PARADA_ANTECIPADA))}
    del matrizes
    classificador, regressor = modelos["vencedor"]
    for nome, modelo in [("Etapa 1 — classificador (P(consumo))", classificador),
                         ("Etapa 2 — regressor (tamanho)", regressor)]:
        print(f"{nome}: melhor rodada {modelo.best_iteration:,}")
        print(importancia(modelo).to_string())

    # features só da confiabilidade: depois da busca, para não entrar no registro nem no modelo de previsão
    df = criar_features_confiabilidade(df, CHAVE_SERIE, JANELA_INTERVALOS, JANELA_TAMANHO, JANELA_PICOS,
                                       coluna=COLUNA_FEATURES, tipo=TIPO_FEATURES)
    linhas_teste = (base == "teste").fillna(False).to_numpy()
    teste = df.loc[linhas_teste,
                   [*CHAVE_SERIE, "Data", "Consumo", COLUNA_ALVO, "EhPico", "Descripcion", "Alvo", *BASELINES, *FEATURES,
                    *FEATURES_SERIE_CONFIABILIDADE]]
    teste = teste.loc[:, ~teste.columns.duplicated()].copy()
    teste["Previsao"], teste["Probabilidade"] = prever_hurdle(classificador, regressor, df, FEATURES,
                                                             PREFIXOS_ESCALAVEIS, escala, linhas_teste)
    # parâmetros de settings como concorrente: mostra o ganho da busca no teste
    padrao = ("PrevisaoPadrao",) if "padrao" in modelos else ()
    if padrao:
        teste["PrevisaoPadrao"] = prever_hurdle(*modelos["padrao"], df, FEATURES, PREFIXOS_ESCALAVEIS, escala,
                                                linhas_teste)[0]
    inicio = etapa("treinamento", inicio)

    concorrentes = ("Previsao", *padrao, *BASELINES)
    comuns = teste[list(BASELINES)].notna().all(axis=1)
    por_serie = metricas_por_serie(teste[comuns], CHAVE_SERIE, concorrentes)
    print(f"Teste — linhas com todos os baselines ({comuns.sum():,} de {len(teste):,}):")
    print(comparar_modelos(por_serie, CHAVE_SERIE, "Previsao").to_string(float_format="{:,.4f}".format))
    # segmentos ABC pelo volume antes do teste
    segmentos = segmentar_abc(df, CHAVE_SERIE, COLUNA_ALVO, (df["Data"] < inicio_da_base(df, linhas_teste)).to_numpy())
    print("Teste — WAPE mediano por série em cada segmento ABC (volume antes do teste: A = 80%, B = 80–95%):")
    print(comparar_por_segmento(por_serie, CHAVE_SERIE, segmentos).to_string(float_format="{:,.4f}".format))
    if len(clientes) > 1:
        for cliente in clientes:
            parte = por_serie[por_serie["Cliente"] == cliente]
            print(f"Teste — {cliente}:")
            print(comparar_modelos(parte, CHAVE_SERIE, "Previsao").to_string(float_format="{:,.4f}".format))
            print(comparar_por_segmento(parte, CHAVE_SERIE, segmentos).to_string(float_format="{:,.4f}".format))
    por_serie = por_serie.join(segmentos, on=list(CHAVE_SERIE))
    por_serie["Segmento"] = por_serie["Segmento"].fillna("C")
    print("Teste — modelo em todas as linhas:")
    print(comparar_modelos(metricas_por_serie(teste, CHAVE_SERIE, ("Previsao",)), CHAVE_SERIE, "Previsao")
          .to_string(float_format="{:,.4f}".format))

    inicio = etapa("métricas", inicio)
    teste, relatorio, ajuste = classificar_confiabilidade(df, base, classificador, regressor, escala, teste, segmentos)
    inicio = etapa("confiabilidade", inicio)

    PASTA_MODELO.mkdir(parents=True, exist_ok=True)
    gravar_confiabilidade(ajuste, PASTA_MODELO)
    relatorio.reset_index().to_csv(PASTA_MODELO / "metricas_confiabilidade.csv", sep=";", decimal=",", index=False,
                                   encoding="utf-8-sig")
    classificador.save_model(PASTA_MODELO / "hurdle_classificador.txt", num_iteration=classificador.best_iteration)
    regressor.save_model(PASTA_MODELO / "hurdle_regressor.txt", num_iteration=regressor.best_iteration)
    teste.to_parquet(PASTA_MODELO / "teste_previsoes.parquet", index=False)
    por_serie.to_parquet(PASTA_MODELO / "metricas_por_serie.parquet", index=False)
    (PASTA_MODELO / "hiperparametros.json").write_text(json.dumps(busca, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo, hiperparâmetros, previsões do teste, confiabilidade e métricas por série em {PASTA_MODELO}")
    etapa("gravação", inicio)


def main() -> None:
    caminho_base = PASTA_FEATURES / f"{nome_base(CLIENTES)}.parquet"
    if REAPROVEITAR_BASE and caminho_base.exists():
        inicio = time.time()
        df = pd.read_parquet(caminho_base)
        etapa(f"base salva lida de {caminho_base} ({len(df):,} linhas)", inicio)
    else:
        df = montar_base()
        PASTA_FEATURES.mkdir(parents=True, exist_ok=True)
        df.to_parquet(caminho_base, index=False, compression="zstd")
        print(f"Base gravada em {caminho_base}")
    treinar(df)


if __name__ == "__main__":
    main()
