# Busca de hiperparâmetros: sorteio reprodutível, registro retomável, etapas separadas = hurdle original,
# segmentação ABC e escolha só pela validação.
# Rodar: .venv/Scripts/python.exe tests/test_busca.py  (ou pytest, se instalado)

import json
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ddmrp_forecast.config.settings import (  # noqa: E402
    CHAVE_SERIE, ESPACO_BUSCA, ESPACO_BUSCA_REGRESSOR, PARAMETROS_CLASSIFICADOR, PARAMETROS_REGRESSOR,
    PREFIXOS_ESCALAVEIS,
)
from ddmrp_forecast.features.defasagens import criar_lags, criar_medias_moveis  # noqa: E402
from ddmrp_forecast.features.intermitencia import criar_frequencias, criar_recencia, criar_tsb_sba  # noqa: E402
from ddmrp_forecast.model.alvo import criar_alvo  # noqa: E402
from ddmrp_forecast.model.busca import (  # noqa: E402
    executar_busca, gravar_tentativa, identificar_base, ler_registro, sortear_parametros,
)
from ddmrp_forecast.model.hurdle import (  # noqa: E402
    calcular_escala, carregar_hurdle, matriz_normalizada, montar_matrizes, prever_hurdle, treinar_hurdle,
)
from ddmrp_forecast.model.metricas import comparar_por_segmento, medianas_por_segmento, segmentar_abc  # noqa: E402
from ddmrp_forecast.model.split import datas_corte, separar_bases  # noqa: E402
from test_modelo import base_diaria  # noqa: E402

FEATURES = ("Lag_1", "Lag_7", "Media_28", "Media_90", "Freq_28", "Freq_90", "DiasDesdeVenda", "TSB", "SBA")
ESCALA = ("Media_365", "Media_90")
PEQUENO = {"min_data_in_leaf": 20, "num_leaves": 15}


def base_sintetica(series: int = 40, dias: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    gerador = np.random.default_rng(2)
    consumos = {}
    for i in range(series):
        tamanho, chance = 10 ** gerador.uniform(0, 3), gerador.uniform(0.05, 0.6)
        venda = gerador.random(dias) < chance
        consumos[f"R{i}"] = np.where(venda, gerador.gamma(4.0, tamanho / 4.0, dias), 0).tolist()
    consumos["MORTA"] = [0.0] * dias
    df = base_diaria(consumos)
    df = criar_lags(df, CHAVE_SERIE, (1, 7))
    df = criar_medias_moveis(df, CHAVE_SERIE, (28, 90, 365))
    df = criar_frequencias(df, CHAVE_SERIE, (28, 90))
    df = criar_recencia(df, CHAVE_SERIE)
    df = criar_tsb_sba(df, CHAVE_SERIE)
    df = criar_alvo(df, CHAVE_SERIE, "Consumo_Limpo", 7)
    inicio_validacao, inicio_teste = datas_corte(df["Data"].max(), 60, 40)
    return df, separar_bases(df, 7, inicio_validacao, inicio_teste)


def test_sorteio_reprodutivel():
    assert sortear_parametros(ESPACO_BUSCA_REGRESSOR, 7, 0) == {}
    for indice in range(1, 200):
        sorteio = sortear_parametros(ESPACO_BUSCA_REGRESSOR, 7, indice)
        assert sorteio == sortear_parametros(ESPACO_BUSCA_REGRESSOR, 7, indice)
        assert json.loads(json.dumps(sorteio)) == sorteio  # só tipos do Python (vai para o .jsonl)
        for nome, (tipo, *limites) in ESPACO_BUSCA_REGRESSOR.items():
            valor = sorteio[nome]
            if tipo == "escolha":
                assert valor in limites[0], (nome, valor)
            elif nome == "num_leaves" and sorteio["max_depth"] > 0:
                assert valor <= min(limites[1], 2 ** sorteio["max_depth"]), sorteio
            else:
                assert valor == 0 or limites[0] <= valor <= limites[1], (nome, valor)
    assert sortear_parametros(ESPACO_BUSCA, 7, 3) != sortear_parametros(ESPACO_BUSCA, 8, 3)
    assert sortear_parametros(ESPACO_BUSCA, 7, 3) != sortear_parametros(ESPACO_BUSCA, 7, 4)


def test_registro_retomavel():
    with tempfile.TemporaryDirectory() as pasta:
        caminho = Path(pasta) / "busca" / "tentativas.jsonl"
        assert ler_registro(caminho) == {}
        gravar_tentativa(caminho, {"chave": "classificador:0", "fase": "classificador", "metricas": {"LogLoss": 0.3}})
        gravar_tentativa(caminho, {"chave": "classificador:1", "fase": "classificador", "metricas": {"LogLoss": 0.2}})
        with caminho.open("a", encoding="utf-8") as arquivo:
            arquivo.write('{"chave": "classificador:2", "fa')  # interrupção no meio da gravação
        registro = ler_registro(caminho)
        assert sorted(registro) == ["classificador:0", "classificador:1"]
        assert registro["classificador:1"]["metricas"]["LogLoss"] == 0.2


def treinar_hurdle_original(df, base, features, prefixos, escala, parametros_classificador, parametros_regressor,
                            rodadas, parada):
    # cópia da versão de f6a6f66 (antes de separar as etapas)
    com_escala = escala > 0
    y = df["Alvo"].to_numpy(np.float32)
    treino = (base == "treino").fillna(False).to_numpy() & com_escala
    validacao = (base == "validacao").fillna(False).to_numpy() & com_escala

    def ajustar(parametros, linhas_treino, linhas_validacao, transformar):
        dados_treino = lgb.Dataset(matriz_normalizada(df, features, prefixos, escala, linhas_treino),
                                   transformar(linhas_treino))
        dados_validacao = lgb.Dataset(matriz_normalizada(df, features, prefixos, escala, linhas_validacao),
                                      transformar(linhas_validacao), reference=dados_treino)
        return lgb.train(parametros, dados_treino, num_boost_round=rodadas, valid_sets=[dados_validacao],
                         valid_names=["validacao"], callbacks=[lgb.early_stopping(parada, verbose=False)])

    classificador = ajustar(parametros_classificador, treino, validacao,
                            lambda linhas: (y[linhas] > 0).astype(np.float32))
    com_consumo = y > 0
    regressor = ajustar(parametros_regressor, treino & com_consumo, validacao & com_consumo,
                        lambda linhas: np.log(y[linhas] / escala[linhas]))
    return classificador, regressor


def test_etapas_reproduzem_hurdle_original():
    df, base = base_sintetica()
    escala = calcular_escala(df, ESCALA)
    teste = (base == "teste").fillna(False).to_numpy()
    for parametros in (PEQUENO, PEQUENO | {"min_data_in_leaf": 200, "num_leaves": 7}):
        original = treinar_hurdle_original(df, base, FEATURES, PREFIXOS_ESCALAVEIS, escala,
                                           PARAMETROS_CLASSIFICADOR | parametros, PARAMETROS_REGRESSOR | parametros,
                                           300, 30)
        novo = treinar_hurdle(df, base, FEATURES, PREFIXOS_ESCALAVEIS, escala, PARAMETROS_CLASSIFICADOR | parametros,
                              PARAMETROS_REGRESSOR | parametros, 300, 30)
        assert [m.best_iteration for m in original] == [m.best_iteration for m in novo]
        previsto_original = prever_hurdle(*original, df, FEATURES, PREFIXOS_ESCALAVEIS, escala, teste)
        previsto_novo = prever_hurdle(*novo, df, FEATURES, PREFIXOS_ESCALAVEIS, escala, teste)
        for a, b in zip(previsto_original, previsto_novo):
            assert np.array_equal(a, b)


def test_segmentacao_abc():
    df = pd.DataFrame({"Cliente": "c", "IdBodega": 1, "Referencia": ["A1", "A1", "B1", "C1", "C2", "Z"],
                       "Consumo_Limpo": [50.0, 30.0, 15.0, 4.0, 1.0, 0.0]})
    segmentos = segmentar_abc(df, CHAVE_SERIE, "Consumo_Limpo", np.ones(len(df), bool))
    # acumulado antes de cada série: A1 0%, B1 80%, C1 95%, C2 99%, Z 100%
    assert segmentos.xs("A1", level="Referencia").iloc[0] == "A"
    assert segmentos.xs("B1", level="Referencia").iloc[0] == "B"
    assert set(segmentos.xs("C1", level="Referencia")) | set(segmentos.xs("Z", level="Referencia")) == {"C"}
    por_serie = pd.DataFrame({"Cliente": "c", "IdBodega": 1, "Referencia": ["A1", "B1", "C1", "C2", "NOVA"],
                              "WAPE": [0.1, 0.2, 0.5, 0.7, 0.9]})
    assert medianas_por_segmento(por_serie, CHAVE_SERIE, segmentos) == {"A": 0.1, "B": 0.2, "C": 0.7}
    dois_modelos = pd.concat([por_serie.assign(Modelo="M1"), por_serie.assign(Modelo="M2", WAPE=por_serie["WAPE"] * 2)])
    tabela = comparar_por_segmento(dois_modelos, CHAVE_SERIE, segmentos)
    assert list(tabela.columns) == ["WAPEMedianoA (1 séries)", "WAPEMedianoB (1 séries)", "WAPEMedianoC (3 séries)"]
    assert tabela.loc["M1"].tolist() == [0.1, 0.2, 0.7] and tabela.loc["M2"].tolist() == [0.2, 0.4, 1.4]


def test_segmentacao_abc_por_cliente():
    # cliente pequeno tem a sua própria curva: no ABC conjunto ele cairia todo em C
    grande = pd.DataFrame({"Cliente": "g", "IdBodega": 1, "Referencia": ["A1", "B1", "C1"],
                           "Consumo_Limpo": [8000.0, 1500.0, 500.0]})
    pequeno = pd.DataFrame({"Cliente": "p", "IdBodega": 1, "Referencia": ["A1", "B1", "C1", "Z"],
                            "Consumo_Limpo": [80.0, 15.0, 5.0, 0.0]})
    df = pd.concat([grande, pequeno], ignore_index=True)
    segmentos = segmentar_abc(df, CHAVE_SERIE, "Consumo_Limpo", np.ones(len(df), bool))
    for cliente in ("g", "p"):
        parte = segmentos.xs(cliente, level="Cliente").droplevel("IdBodega")
        assert parte[["A1", "B1", "C1"]].tolist() == ["A", "B", "C"]
    assert segmentos.xs("Z", level="Referencia").iloc[0] == "C"
    # cada cliente sozinho dá a mesma segmentação que junto
    sozinho = segmentar_abc(pequeno, CHAVE_SERIE, "Consumo_Limpo", np.ones(len(pequeno), bool))
    assert sozinho.sort_index().equals(segmentos.xs("p", level="Cliente", drop_level=False).sort_index())
    # sem por cliente, o pequeno inteiro cairia em C
    conjunto = segmentar_abc(df.assign(Todos="t").set_index("Todos").reset_index(), ("Todos", "Cliente", "Referencia"),
                             "Consumo_Limpo", np.ones(len(df), bool), por="Todos")
    assert set(conjunto.xs("p", level="Cliente")) == {"C"}


def test_modelo_salvo_preve_igual():
    df, base = base_sintetica()
    escala = calcular_escala(df, ESCALA)
    modelos = treinar_hurdle(df, base, FEATURES, PREFIXOS_ESCALAVEIS, escala, PARAMETROS_CLASSIFICADOR | PEQUENO,
                             PARAMETROS_REGRESSOR | PEQUENO, 300, 30)
    teste = (base == "teste").fillna(False).to_numpy()
    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        # mesma gravação do main: só as árvores até a melhor rodada
        for nome, modelo in zip(("classificador", "regressor"), modelos):
            modelo.save_model(pasta / f"hurdle_{nome}.txt", num_iteration=modelo.best_iteration)
        carregados = carregar_hurdle(pasta)
        for a, b in zip(prever_hurdle(*modelos, df, FEATURES, PREFIXOS_ESCALAVEIS, escala, teste),
                        prever_hurdle(*carregados, df, FEATURES, PREFIXOS_ESCALAVEIS, escala, teste)):
            assert np.allclose(a, b, rtol=1e-6, atol=1e-9)


def rodar_busca(df, base, pasta):
    escala = calcular_escala(df, ESCALA)
    matrizes = montar_matrizes(df, base, FEATURES, PREFIXOS_ESCALAVEIS, escala)
    return executar_busca(df, base, matrizes, CHAVE_SERIE, "Consumo_Limpo", escala, pasta,
                          PARAMETROS_CLASSIFICADOR | PEQUENO, PARAMETROS_REGRESSOR | PEQUENO,
                          ESPACO_BUSCA, ESPACO_BUSCA_REGRESSOR, 4, 5, 11, 2, (1,), 0.02, 200, 20)


def test_busca_escolhe_pela_validacao():
    df, base = base_sintetica()
    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        resultado = rodar_busca(df, base, pasta / "a")
        registro = ler_registro(pasta / "a" / "tentativas.jsonl")
        fases = pd.Series([t["fase"] for t in registro.values()]).value_counts()
        assert fases["classificador"] == 4 and fases["regressor"] == 5, fases
        assert "produto:0x0" in registro and fases["sementes"] >= 1

        # o vencedor é a melhor média (validação) entre os finalistas aceitos pela trava do WAPE global
        finalistas = pd.read_csv(pasta / "a" / "finalistas.csv", sep=";", decimal=",")
        aceitos = finalistas[finalistas["Aceita"]]
        melhor = aceitos.loc[aceitos["WAPEMediano"].idxmin()]
        assert (resultado["classificador"], resultado["regressor"]) == (melhor["classificador"], melhor["regressor"])
        assert json.loads((pasta / "a" / "vencedor.json").read_text(encoding="utf-8"))["regressor"] == resultado["regressor"]
        assert resultado["parametros_regressor"] == PARAMETROS_REGRESSOR | PEQUENO | sortear_parametros(
            ESPACO_BUSCA_REGRESSOR, 11, resultado["regressor"])
        assert resultado["modelos"]["padrao"][0].best_iteration > 0

        # retomar: nada é refeito e o vencedor é o mesmo
        linhas = len((pasta / "a" / "tentativas.jsonl").read_text(encoding="utf-8").splitlines())
        retomado = rodar_busca(df, base, pasta / "a")
        assert len((pasta / "a" / "tentativas.jsonl").read_text(encoding="utf-8").splitlines()) == linhas
        assert (retomado["classificador"], retomado["regressor"]) == (resultado["classificador"], resultado["regressor"])

        # o teste não é lido: bagunçar alvo e features do teste não muda nenhuma métrica do registro
        baguncado = df.copy()
        teste = (base == "teste").fillna(False).to_numpy()
        gerador = np.random.default_rng(0)
        for coluna in ("Alvo", *FEATURES, *ESCALA):
            baguncado.loc[teste, coluna] = gerador.uniform(0, 1e4, teste.sum()).astype(baguncado[coluna].dtype)
        rodar_busca(baguncado, base, pasta / "b")
        registro_b = ler_registro(pasta / "b" / "tentativas.jsonl")
        assert registro.keys() == registro_b.keys()
        for chave, tentativa in registro.items():
            assert tentativa["metricas"] == registro_b[chave]["metricas"], chave

        # a identificação muda com a base e com a configuração
        assert identificar_base(df, base, {"x": 1}) != identificar_base(baguncado, base, {"x": 1})
        assert identificar_base(df, base, {"x": 1}) != identificar_base(df, base, {"x": 2})


if __name__ == "__main__":
    test_sorteio_reprodutivel()
    test_registro_retomavel()
    test_etapas_reproduzem_hurdle_original()
    test_segmentacao_abc()
    test_segmentacao_abc_por_cliente()
    test_modelo_salvo_preve_igual()
    test_busca_escolhe_pela_validacao()
    print("Busca: OK")
