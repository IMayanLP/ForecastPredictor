import time

from ddmrp_forecast.config.settings import (
    CHAVE_SERIE, CLIENTES, COLUNAS_FIXAS, DATEFIRST_SQL, DIAS_ADU_PADRAO, ESCOPO_BUFFER_MESTRE, FATOR_IQR, JANELA_LIMPEZA,
    JANELAS_MEDIA, LAGS, PASTA_RAW, PASTA_SOURCE, QUARTIS_SO_COM_CONSUMO, TAMANHO_LOTE, TIPO_FEATURES,
)
from ddmrp_forecast.features.datas import adicionar_partes_da_data
from ddmrp_forecast.features.defasagens import criar_lags, criar_medias_moveis
from ddmrp_forecast.io.parquet_store import ler_parquet
from ddmrp_forecast.io.conversao import converter_pasta
from ddmrp_forecast.io.validacao import resumir_periodo, validar_consumos
from ddmrp_forecast.preprocess.adu import corrigir_dias_adu, recalcular_adu
from ddmrp_forecast.preprocess.auditoria import auditar_adu, auditar_picos
from ddmrp_forecast.preprocess.calendario import completar_calendario
from ddmrp_forecast.preprocess.picos import limpar_picos, marcar_picos, resumir_picos
from ddmrp_forecast.preprocess.preparacao import preparar_base, verificar_ordenacao


def etapa(nome: str, inicio: float) -> float:
    print(f"[{nome}] {time.time() - inicio:,.0f}s")
    return time.time()


def main() -> None:
    # 1) .rpt/.csv → Parquet (só converte o que for novo ou alterado)
    converter_pasta(PASTA_SOURCE, PASTA_RAW, TAMANHO_LOTE, CLIENTES)

    # 2) leitura e checagem do Parquet
    inicio = time.time()
    df = ler_parquet(PASTA_RAW, clientes=CLIENTES)
    print(f"Registros lidos: {len(df):,} ({df.memory_usage(deep=True).sum() / 1e9:,.2f} GB em memória)")
    print(f"Período: {resumir_periodo(df)}")
    problemas = validar_consumos(df, CHAVE_SERIE)
    for problema in problemas:
        print(f"  ⚠ {problema}")
    if any(p.startswith("Colunas ausentes") for p in problemas):
        raise SystemExit("Base incompleta: corrigir a exportação do SSMS antes de seguir.")
    df, corrigidos = corrigir_dias_adu(df, DIAS_ADU_PADRAO)
    print(f"DiasADU 0 → {DIAS_ADU_PADRAO:g}: {corrigidos:,} registros")
    inicio = etapa("leitura", inicio)

    # 3) buffers mestres + ordenação
    df, removidas = preparar_base(df, ESCOPO_BUFFER_MESTRE)
    print(f"Linhas de componentes removidas: {removidas:,}")
    print(f"Registros após remoção dos componentes: {len(df):,}")
    verificar_ordenacao(df, CHAVE_SERIE)
    inicio = etapa("buffers mestres", inicio)

    # 4) picos
    df, _ = marcar_picos(df, CHAVE_SERIE, FATOR_IQR, QUARTIS_SO_COM_CONSUMO)
    print(df["EhPico"].value_counts().to_string())
    print(resumir_picos(df, CHAVE_SERIE).head(20).to_string())
    df = limpar_picos(df, CHAVE_SERIE, JANELA_LIMPEZA)
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
    df = criar_lags(df, CHAVE_SERIE, LAGS, tipo=TIPO_FEATURES)
    df = criar_medias_moveis(df, CHAVE_SERIE, JANELAS_MEDIA, tipo=TIPO_FEATURES)
    df, divergencias = adicionar_partes_da_data(df, DATEFIRST_SQL)
    for coluna, quantidade in divergencias.items():
        if quantidade:
            print(f"  ⚠ {coluna}: {quantidade:,} valores exportados diferem do calculado (conferir DATEFIRST_SQL)")
    etapa("features", inicio)

    # 8) auditoria da base final
    verificar_ordenacao(df, CHAVE_SERIE)
    print(f"Ordenação por {[*CHAVE_SERIE, 'Data']}: ok")
    total_erros = 0
    for nome, (informacoes, erros) in [("Picos", auditar_picos(df, JANELA_LIMPEZA)),
                                       ("ADU", auditar_adu(df, CHAVE_SERIE))]:
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


if __name__ == "__main__":
    main()
