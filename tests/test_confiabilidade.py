# Confiabilidade da previsão: features novas × laço de referência, sem vazamento, definição de acerto, classes,
# ajuste com dobras por série, linhas sem escala, regras simples e relatórios.
# Rodar: .venv/Scripts/python.exe tests/test_confiabilidade.py  (ou pytest, se instalado)

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ddmrp_forecast.config.settings import (  # noqa: E402
    CHAVE_SERIE, CORTE_TIPO_PREVISAO, CORTES_CLASSES, FAIXAS_REGRAS, FEATURES_CONFIABILIDADE, FEATURES_SERIE_CONFIABILIDADE,
    PARAMETROS_CONFIABILIDADE, ROTULOS_CLASSES,
)
from ddmrp_forecast.features.intermitencia import criar_features_confiabilidade  # noqa: E402
from ddmrp_forecast.model.confiabilidade import (  # noqa: E402
    ajustar_confiabilidade, aplicar_confiabilidade, bootstrap_por_serie, carregar_confiabilidade, classificar,
    comparar_com_regras, dobras_por_serie, eh_monotona, gravar_confiabilidade, marcar_acerto, preparar_sinais,
    relatorio_por_recorte, tabela_por_classe, taxa_da_regra, tipo_previsao,
)
from test_modelo import base_diaria  # noqa: E402

PEQUENO = PARAMETROS_CONFIABILIDADE | {"min_data_in_leaf": 20}


def referencia_features(valores: list[float], picos: list[bool], intervalos: int, tamanho: int,
                        janela_picos: int) -> dict[str, list[float]]:
    saida = {"ADI": [], "CV2Intervalo": [], "CV2Tamanho": [], "Picos": []}
    for t in range(len(valores)):
        vendas = [i for i in range(t) if valores[i] > 0]
        diferencas = [b - a for a, b in zip(vendas, vendas[1:])]
        # a janela corre sobre as vendas: as `intervalos` últimas podem incluir a 1ª, que não tem intervalo
        recentes = diferencas[-intervalos:] if len(vendas) > intervalos else diferencas
        saida["ADI"].append(np.mean(recentes) if recentes else np.nan)
        saida["CV2Intervalo"].append((np.std(recentes, ddof=1) / np.mean(recentes)) ** 2 if len(recentes) >= 2 else np.nan)
        janela = [v for v in valores[max(0, t - tamanho):t] if v > 0]
        saida["CV2Tamanho"].append(np.var(janela) / np.mean(janela) ** 2 if len(janela) >= 2 else np.nan)
        saida["Picos"].append(sum(picos[max(0, t - janela_picos):t]) if t > 0 else np.nan)
    return saida


def test_features_iguais_a_referencia():
    gerador = np.random.default_rng(3)
    consumos = {f"R{i}": np.where(gerador.random(200) < chance, gerador.integers(1, 50, 200), 0).astype(float).tolist()
                for i, chance in enumerate((0.25, 0.05, 0.6))}
    consumos["R0"][:15] = [0.0] * 15                      # começa sem venda
    consumos["MORTA"] = [0.0] * 50                        # nunca vende
    df = base_diaria(consumos)
    df["EhPico"] = pd.Series(gerador.random(len(df)) < 0.03, dtype=object)
    df.loc[df.index[5], "EhPico"] = None                  # dia acrescentado no calendário não tem marcação
    df = criar_features_confiabilidade(df, CHAVE_SERIE, 5, 30, 20)
    for ref, valores in consumos.items():
        linhas = df["Referencia"] == ref
        esperado = referencia_features(valores, df.loc[linhas, "EhPico"].eq(True).tolist(), 5, 30, 20)
        np.testing.assert_allclose(df.loc[linhas, "ADI_5"], esperado["ADI"], rtol=1e-5)
        np.testing.assert_allclose(df.loc[linhas, "CV2Intervalo_5"], esperado["CV2Intervalo"], rtol=1e-4)
        np.testing.assert_allclose(df.loc[linhas, "CV2Tamanho_30"], esperado["CV2Tamanho"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(df.loc[linhas, "PicosRecentes_20"], esperado["Picos"])

    # sem vazamento: mudar o consumo e o pico do dia 100 não altera nenhuma feature até o dia 100
    alterado = base_diaria({**consumos, "R1": consumos["R1"][:100] + [999.0] + consumos["R1"][101:]})
    alterado["EhPico"] = df["EhPico"].to_numpy()
    posicao_100 = df.index[df["Referencia"] == "R1"][100]
    alterado.loc[posicao_100, "EhPico"] = not bool(alterado.loc[posicao_100, "EhPico"])
    alterado = criar_features_confiabilidade(alterado, CHAVE_SERIE, 5, 30, 20)
    colunas = ["ADI_5", "CV2Intervalo_5", "CV2Tamanho_30", "PicosRecentes_20"]
    ate_100 = (df["Referencia"] == "R1") & (df["Data"] <= df["Data"].min() + pd.Timedelta(days=100))
    pd.testing.assert_frame_equal(df.loc[ate_100, colunas], alterado.loc[ate_100, colunas])
    # o dia seguinte já enxerga a mudança
    dia_101 = df.index[df["Referencia"] == "R1"][101]
    assert df.loc[dia_101, "PicosRecentes_20"] != alterado.loc[dia_101, "PicosRecentes_20"]


def test_acerto_feito_a_mao():
    escala = np.full(7, 10.0)
    alvo = np.array([0.0, 0.0, 10.0, 10.0, 40.0, 40.0, 2.0])
    previsao = np.array([3.0, 3.1, 13.0, 6.9, 52.0, 27.9, 5.0])
    esperado = [True, False, True, False, True, False, True]
    assert marcar_acerto(alvo, previsao, escala, 0.3).tolist() == esperado
    # sem escala, previsão 0: só acerta quem de fato não vendeu
    assert marcar_acerto(np.array([0.0, 5.0]), np.zeros(2), np.zeros(2), 0.3).tolist() == [True, False]


def test_classes_e_tipo():
    classes = classificar(np.array([0.0, 0.1999, 0.2, 0.4, 0.5999, 0.6, 0.8, 1.0]), CORTES_CLASSES, ROTULOS_CLASSES)
    assert list(classes) == ["Muito baixa", "Muito baixa", "Baixa", "Média", "Média", "Alta", "Muito alta", "Muito alta"]
    assert classes.ordered and classes.min() == "Muito baixa"
    assert tipo_previsao(np.array([0.49, 0.5, 0.9]), CORTE_TIPO_PREVISAO).tolist() == ["Sem consumo", "Com consumo", "Com consumo"]


def linhas_sinteticas(series: int, dias: int, semente: int, sem_escala: int = 0) -> pd.DataFrame:
    # o erro da previsão depende de Freq_90: quanto mais frequente a série, mais a previsão acerta
    gerador = np.random.default_rng(semente)
    partes = []
    for i in range(series):
        frequencia, escala = gerador.uniform(0, 1), 10 ** gerador.uniform(0, 3)
        ruido = (1.2 - frequencia) * gerador.normal(0, 0.6, dias)
        alvo = np.maximum(escala * (1 + ruido), 0)
        partes.append(pd.DataFrame({
            "Cliente": "K", "Referencia": f"R{i}", "IdBodega": "1", "Data": pd.date_range("2026-01-01", periods=dias),
            "Alvo": alvo, "Freq_28": frequencia + gerador.normal(0, 0.05, dias), "Freq_90": frequencia,
            "Freq_365": frequencia, "DiasDesdeVenda": gerador.integers(1, 30, dias).astype(float),
            "DiasHistorico": 500.0, "ADI_12": 1 / max(frequencia, 0.01), "CV2Intervalo_12": 0.5, "CV2Tamanho_365": 0.5,
            "PicosRecentes_90": 0.0, "_Previsao": escala, "_Probabilidade": 0.9, "_Escala": escala,
        }))
    for i in range(sem_escala):
        partes.append(pd.DataFrame({
            "Cliente": "K", "Referencia": f"Z{i}", "IdBodega": "1", "Data": pd.date_range("2026-01-01", periods=dias),
            "Alvo": np.where(gerador.random(dias) < 0.25, 3.0, 0.0), "Freq_28": 0.0, "Freq_90": 0.0, "Freq_365": 0.0,
            "DiasDesdeVenda": np.nan, "DiasHistorico": 500.0, "ADI_12": np.nan, "CV2Intervalo_12": np.nan,
            "CV2Tamanho_365": np.nan, "PicosRecentes_90": 0.0, "_Previsao": 0.0, "_Probabilidade": 0.0, "_Escala": 0.0,
        }))
    linhas = pd.concat(partes, ignore_index=True)
    return preparar_sinais(linhas, linhas.pop("_Previsao").to_numpy(), linhas.pop("_Probabilidade").to_numpy(),
                           linhas.pop("_Escala").to_numpy())


def teste_classificado(ajuste: dict, teste: pd.DataFrame) -> pd.DataFrame:
    teste["Confiabilidade"] = aplicar_confiabilidade(teste, ajuste)
    teste["ClasseConfiabilidade"] = classificar(teste["Confiabilidade"].to_numpy(), CORTES_CLASSES, ROTULOS_CLASSES)
    teste["TipoPrevisao"] = tipo_previsao(teste["Probabilidade"].to_numpy(), CORTE_TIPO_PREVISAO)
    teste["Acerto"] = marcar_acerto(teste["Alvo"].to_numpy(), teste["Previsao"].to_numpy(), teste["Escala"].to_numpy(), 0.3)
    return teste


def test_ajuste_separa_e_calibra():
    validacao, teste = linhas_sinteticas(200, 30, 1, sem_escala=10), linhas_sinteticas(200, 30, 2, sem_escala=10)
    codigos = dobras_por_serie(validacao, CHAVE_SERIE, 5)
    assert (pd.Series(codigos).groupby(validacao["Referencia"].to_numpy()).nunique() == 1).all()
    assert pd.Series(codigos).nunique() == 5

    ajuste = ajustar_confiabilidade(validacao, CHAVE_SERIE, FEATURES_CONFIABILIDADE, 0.3, PEQUENO, 5, 300, 30)
    teste = teste_classificado(ajuste, teste)
    com_escala = teste[teste["Escala"] > 0]
    tabela = tabela_por_classe(com_escala, CHAVE_SERIE)
    assert eh_monotona(tabela, 100), tabela
    assert (tabela["Linhas"] > 0).sum() >= 3, tabela
    # calibração: previsto próximo do real em cada classe com volume
    com_volume = tabela[tabela["Linhas"] >= 100]
    assert (com_volume["Estimada"] - com_volume["Acerto"]).abs().max() < 0.1, tabela
    assert com_escala["Confiabilidade"].between(0, 1).all()
    assert np.corrcoef(com_escala["Confiabilidade"], com_escala["Freq_90"])[0, 1] > 0.5

    # sem escala não passa pelo modelo: recebe a taxa de acerto da validação (~75%, quem não vendeu)
    sem_escala = validacao["Escala"] == 0
    assert np.isclose(ajuste["taxa_sem_escala"], (validacao.loc[sem_escala, "Alvo"] == 0).mean())
    assert (teste.loc[teste["Escala"] == 0, "Confiabilidade"] == ajuste["taxa_sem_escala"]).all()

    # a confiabilidade não enxerga o alvo do período aplicado, e funciona em origens sem alvo
    sem_alvo = teste.drop(columns=["Alvo", "Acerto"])
    np.testing.assert_array_equal(aplicar_confiabilidade(sem_alvo, ajuste), teste["Confiabilidade"].to_numpy())

    # gravar e carregar: mesma %
    with tempfile.TemporaryDirectory() as pasta:
        gravar_confiabilidade(ajuste, Path(pasta))
        carregado = carregar_confiabilidade(Path(pasta))
        np.testing.assert_allclose(aplicar_confiabilidade(teste, carregado), teste["Confiabilidade"], rtol=1e-9)
        del carregado["modelo"]

    # relatório: recortes, sem linhas repetidas e bootstrap por série
    teste["Segmento"] = np.where(teste["Referencia"].str[1:].astype(int) < 20, "A", "C")
    relatorio = relatorio_por_recorte(teste, CHAVE_SERIE)
    linhas_por_recorte = relatorio.groupby(level="Recorte", sort=False)["Linhas"].sum()
    assert linhas_por_recorte["Todas"] == len(teste)
    assert linhas_por_recorte["Com escala"] + linhas_por_recorte["Sem escala"] == len(teste)
    assert linhas_por_recorte["Segmento A"] + linhas_por_recorte["Segmento C"] == linhas_por_recorte["Com escala"]
    assert not relatorio.index.get_level_values("Recorte").str.startswith("Cliente").any()
    intervalo, monotonos = bootstrap_por_serie(com_escala, CHAVE_SERIE, 50, 1, 100)
    assert 0.9 <= monotonos <= 1 and (intervalo["Acerto_p5"] <= intervalo["Acerto_p95"]).all()

    # modelo × regras: mesmas linhas para todos, e o modelo bate a regra que não tem o sinal
    validacao = teste_classificado(ajuste, validacao)
    validacao["Segmento"] = "C"
    filtro = (teste["Escala"] > 0) & (teste["Segmento"] == "C")
    regras = comparar_com_regras(validacao[validacao["Escala"] > 0], teste[filtro], {"Só ABC": ("Segmento",),
                                 "Só Freq_90": ("Freq_90",)}, FAIXAS_REGRAS, CORTES_CLASSES, ROTULOS_CLASSES)
    assert regras.loc["Só ABC", "AUC"] == 0.5
    assert regras.loc["Modelo", "AUC"] > regras.loc["Só ABC", "AUC"]
    assert list(regras.columns[-5:]) == list(ROTULOS_CLASSES)


def test_tabela_e_regra_feitas_a_mao():
    linhas = pd.DataFrame({
        "Cliente": "K", "Referencia": ["R1", "R1", "R2", "R2"], "IdBodega": "1", "Alvo": [10.0, 0.0, 10.0, 20.0],
        "Previsao": [9.0, 1.0, 5.0, 25.0], "Acerto": [True, False, False, True], "Confiabilidade": [0.1, 0.1, 0.9, 0.7],
    })
    linhas["ClasseConfiabilidade"] = classificar(linhas["Confiabilidade"].to_numpy(), CORTES_CLASSES, ROTULOS_CLASSES)
    tabela = tabela_por_classe(linhas, CHAVE_SERIE)
    baixa, alta = tabela.loc["Muito baixa"], tabela.loc["Muito alta"]
    assert baixa["Linhas"] == 2 and baixa["Acerto"] == 0.5 and baixa["AlvoZero"] == 0.5 and baixa["Distribuicao"] == 0.5
    assert np.isclose(baixa["WAPE"], 2 / 10) and np.isclose(baixa["WAPEMediano"], 2 / 10) and baixa["Series"] == 1
    assert alta["Linhas"] == 1 and np.isclose(alta["WAPE"], 0.5)
    assert tabela.loc["Média", "Linhas"] == 0 and np.isnan(tabela.loc["Média", "Acerto"])
    assert list(tabela.index) == list(ROTULOS_CLASSES)

    assert eh_monotona(pd.DataFrame({"Linhas": [500, 500, 500], "Acerto": [0.1, 0.5, 0.9]}), 100)
    assert not eh_monotona(pd.DataFrame({"Linhas": [500, 500, 500], "Acerto": [0.1, 0.9, 0.5]}), 100)
    # classe com poucas linhas não conta
    assert eh_monotona(pd.DataFrame({"Linhas": [500, 5, 500], "Acerto": [0.1, 0.0, 0.5]}), 100)

    validacao = pd.DataFrame({"Freq_90": [0.01, 0.01, 0.5, 0.5, np.nan], "Acerto": [False, False, True, False, False]})
    teste = pd.DataFrame({"Freq_90": [0.02, 0.3, np.nan]})
    faixas = {"Freq_90": ((0.02, 0.05, 0.1, 0.25), -1.0)}
    np.testing.assert_allclose(taxa_da_regra(validacao, teste, ("Freq_90",), faixas), [0.0, 0.5, 0.0])
    # faixa que não existia na validação recebe a taxa geral
    novo = pd.DataFrame({"Freq_90": [0.07]})
    np.testing.assert_allclose(taxa_da_regra(validacao, novo, ("Freq_90",), faixas), [0.2])


def test_features_fora_do_modelo_de_previsao():
    from ddmrp_forecast.config.settings import FEATURES
    assert not set(FEATURES_SERIE_CONFIABILIDADE) & set(FEATURES)
    assert not {"Probabilidade", "PrevSobreEscala", "LogEscala", "Segmento"} & set(FEATURES)
    assert "LogEscala" not in FEATURES_CONFIABILIDADE and "Segmento" not in FEATURES_CONFIABILIDADE


if __name__ == "__main__":
    test_features_iguais_a_referencia()
    test_acerto_feito_a_mao()
    test_classes_e_tipo()
    test_ajuste_separa_e_calibra()
    test_tabela_e_regra_feitas_a_mao()
    test_features_fora_do_modelo_de_previsao()
    print("Confiabilidade: OK")
