# Alvo à frente, split temporal sem vazamento, quartis só do treino, métricas e treino rápido.
# Rodar: .venv/Scripts/python.exe tests/test_modelo.py  (ou pytest, se instalado)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddmrp_forecast.config.settings import (  # noqa: E402
    CHAVE_SERIE, PARAMETROS_CLASSIFICADOR, PARAMETROS_LGBM, PARAMETROS_REGRESSOR, PREFIXOS_ESCALAVEIS,
)
from ddmrp_forecast.features.defasagens import criar_lags, criar_medias_moveis  # noqa: E402
from ddmrp_forecast.features.intermitencia import criar_frequencias, criar_recencia, criar_tsb_sba  # noqa: E402
from ddmrp_forecast.model.hurdle import calcular_escala, prever_hurdle, treinar_hurdle  # noqa: E402
from ddmrp_forecast.model.alvo import criar_alvo  # noqa: E402
from ddmrp_forecast.model.metricas import comparar_modelos, metricas_por_serie  # noqa: E402
from ddmrp_forecast.model.split import UM_DIA, datas_corte, separar_bases  # noqa: E402
from ddmrp_forecast.model.treino import prever, treinar_lgbm  # noqa: E402
from ddmrp_forecast.preprocess.adu import recalcular_adu  # noqa: E402
from ddmrp_forecast.preprocess.auditoria import auditar_features, auditar_picos  # noqa: E402
from ddmrp_forecast.preprocess.calendario import completar_calendario  # noqa: E402
from ddmrp_forecast.preprocess.picos import limpar_picos, marcar_picos  # noqa: E402
from ddmrp_forecast.preprocess.preparacao import remover_sob_encomenda  # noqa: E402


def base_diaria(consumos: dict[str, list[float]], inicio: str = "2026-01-01") -> pd.DataFrame:
    partes = [
        pd.DataFrame({"Cliente": "K", "Referencia": ref, "IdBodega": "1",
                      "Data": pd.date_range(inicio, periods=len(valores)), "Consumo_Limpo": valores})
        for ref, valores in consumos.items()
    ]
    return pd.concat(partes, ignore_index=True)


def test_alvo_a_frente():
    df = criar_alvo(base_diaria({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30]}), CHAVE_SERIE, "Consumo_Limpo", 3)
    esperado = [2.0, 3.0, 4.0, np.nan, np.nan, 20.0, np.nan, np.nan]   # B não usa valores de A
    np.testing.assert_allclose(df["Alvo"].to_numpy(), esperado)

    df = criar_alvo(df, CHAVE_SERIE, "Consumo_Limpo", 1)
    np.testing.assert_allclose(df["Alvo"], df["Consumo_Limpo"])


def test_split_sem_sobreposicao():
    horizonte = 7
    df = criar_alvo(base_diaria({"A": list(range(200)), "B": [1.0] * 150}), CHAVE_SERIE, "Consumo_Limpo", horizonte)
    inicio_validacao, inicio_teste = datas_corte(df["Data"].max(), dias_teste=40, dias_validacao=30)
    assert inicio_teste == df["Data"].max() - pd.Timedelta(days=39)
    assert inicio_validacao == inicio_teste - pd.Timedelta(days=30)

    base = separar_bases(df, horizonte, inicio_validacao, inicio_teste)
    fim_janela = df["Data"] + (horizonte - 1) * UM_DIA
    assert (fim_janela[base == "treino"] < inicio_validacao).all()
    assert (df["Data"][base == "validacao"] >= inicio_validacao).all()
    assert (fim_janela[base == "validacao"] < inicio_teste).all()
    assert (df["Data"][base == "teste"] >= inicio_teste).all()
    assert df.loc[base.notna(), "Alvo"].notna().all()
    assert base.isna()[df["Alvo"].isna()].all()
    # A: 34 dias de teste com janela completa (40 - 7 + 1); B acaba antes do teste
    assert (base[df["Referencia"] == "A"] == "teste").sum() == 34
    assert (base[df["Referencia"] == "B"] == "teste").sum() == 0


def test_quartis_so_do_treino():
    # vendas normais até o corte; depois do corte só vendas altas, que não podem mexer no limite
    consumo = [5.0, 6.0, 5.0, 7.0, 6.0, 5.0] + [100.0, 120.0, 110.0, 90.0]
    df = base_diaria({"A": consumo}).rename(columns={"Consumo_Limpo": "Consumo"})
    corte = df["Data"].iloc[6]

    todo, quartis_todo = marcar_picos(df.copy(), CHAVE_SERIE, 3.0, True)
    treino, quartis_treino = marcar_picos(df.copy(), CHAVE_SERIE, 3.0, True, ate=corte)
    assert quartis_treino.loc[0, ["Q1", "Q3"]].tolist() == [5.0, 6.0]
    assert quartis_todo.loc[0, "Q3"] > 6.0
    assert treino["EhPico"].tolist() == [False] * 6 + [True] * 4
    assert not todo["EhPico"].any()


def test_adu_sobre_consumo_limpo():
    df = base_diaria({"A": [0.0, 4.0, 2.0, 0.0, 6.0], "B": [3.0, 3.0, 9.0]})
    df["Consumo"] = df["Consumo_Limpo"] * 10
    df["DiasADU"] = [2.0] * 5 + [1.9] * 3          # 1,9 trunca para 1, como no ADU_recalculado
    df = recalcular_adu(df, CHAVE_SERIE)
    df = recalcular_adu(df, CHAVE_SERIE, coluna="Consumo_Limpo", nome="ADU_Limpo")
    np.testing.assert_allclose(df["ADU_Limpo"], [np.nan, np.nan, 2.0, 3.0, 1.0, np.nan, 3.0, 3.0])
    np.testing.assert_allclose(df["ADU_recalculado"], df["ADU_Limpo"] * 10)


def test_features_sem_picos():
    # pico de 500 no dia 10: nenhum lag/média pode enxergá-lo
    consumo = [5.0, 6.0, 5.0, 7.0, 6.0, 5.0, 6.0, 5.0, 7.0, 6.0, 500.0, 5.0, 6.0, 5.0, 7.0]
    df = base_diaria({"A": consumo}).rename(columns={"Consumo_Limpo": "Consumo"})
    df, _ = marcar_picos(df, CHAVE_SERIE, 3.0, True)
    df = limpar_picos(df, CHAVE_SERIE, 7)
    df, _ = completar_calendario(df, CHAVE_SERIE, [])
    assert df["EhPico"].tolist() == [False] * 10 + [True] + [False] * 4

    lags, janelas = (1, 2), (3,)
    limpo = criar_medias_moveis(criar_lags(df.copy(), CHAVE_SERIE, lags), CHAVE_SERIE, janelas)
    assert limpo[["Lag_1", "Lag_2", "Media_3"]].max().max() < 10
    assert limpo.loc[11, "Lag_1"] == np.float32(np.mean([7.0, 6.0, 5.0, 6.0, 5.0, 7.0, 6.0]))   # média dos 7 dias antes do pico
    informacoes, erros = auditar_features(limpo, CHAVE_SERIE, lags, janelas)
    assert sum(erros.values()) == 0
    assert informacoes["Lag_1 ≠ Consumo bruto (picos excluídos)"] == 1

    bruto = criar_medias_moveis(criar_lags(df.copy(), CHAVE_SERIE, lags, coluna="Consumo"), CHAVE_SERIE, janelas,
                                coluna="Consumo")
    _, erros = auditar_features(bruto, CHAVE_SERIE, lags, janelas)
    assert erros["Lag_1 diferente de Consumo_Limpo"] == 1 and erros["Media_3 diferente de Consumo_Limpo"] == 3


def referencia_intermitencia(valores: list[float], alfa: float, janela: int) -> dict[str, list[float]]:
    # laço simples, dia a dia: a saída em t só usa valores até t-1
    saida = {"Freq": [], "DiasDesdeVenda": [], "TSB": [], "SBA": []}
    prob = tamanho_tsb = tamanho_sba = intervalo = ultima = None
    for t, q in enumerate(valores):
        anteriores = valores[max(0, t - janela):t]
        saida["Freq"].append(np.mean([v > 0 for v in anteriores]) if t >= janela else np.nan)
        saida["DiasDesdeVenda"].append(t - ultima if ultima is not None else np.nan)
        saida["TSB"].append(prob * tamanho_tsb if prob is not None and tamanho_tsb is not None else 0.0)
        saida["SBA"].append((1 - alfa / 2) * tamanho_sba / intervalo if tamanho_sba is not None else np.nan)
        venda = q > 0
        prob = float(venda) if prob is None else alfa * venda + (1 - alfa) * prob
        if venda:
            novo_intervalo = t - ultima if ultima is not None else t + 1
            tamanho_tsb = q if tamanho_tsb is None else alfa * q + (1 - alfa) * tamanho_tsb
            tamanho_sba = q if tamanho_sba is None else alfa * q + (1 - alfa) * tamanho_sba
            intervalo = novo_intervalo if intervalo is None else alfa * novo_intervalo + (1 - alfa) * intervalo
            ultima = t
    return saida


def test_pico_vira_limite_superior():
    # mesma série do teste acima: limite = 6,5 + 3·1,5 = 11; o pico de 500 vira 11, o resto fica igual
    consumo = [5.0, 6.0, 5.0, 7.0, 6.0, 5.0, 6.0, 5.0, 7.0, 6.0, 500.0, 5.0, 6.0, 5.0, 7.0]
    df = base_diaria({"A": consumo}).rename(columns={"Consumo_Limpo": "Consumo"})
    df, _ = marcar_picos(df, CHAVE_SERIE, 3.0, True)
    df = limpar_picos(df, CHAVE_SERIE, 7, substituicao="limite")
    assert df["Consumo_Limpo"].tolist() == consumo[:10] + [11.0] + consumo[11:]
    assert "Media_7_Limpa" not in df.columns

    informacoes, erros = auditar_picos(df, 7, "limite")
    assert sum(erros.values()) == 0 and informacoes["picos"] == 1
    df.loc[10, "Consumo_Limpo"] = 6.0                 # a regra da média não passa na auditoria do limite
    assert auditar_picos(df, 7, "limite")[1]["Consumo_Limpo diferente do esperado"] == 1

    # série sem nenhuma venda: sem limite e sem pico, nada muda
    vazia = base_diaria({"Z": [0.0] * 5}).rename(columns={"Consumo_Limpo": "Consumo"})
    vazia, _ = marcar_picos(vazia, CHAVE_SERIE, 3.0, True)
    assert limpar_picos(vazia, CHAVE_SERIE, 7, "limite")["Consumo_Limpo"].tolist() == [0.0] * 5


def test_intermitencia_igual_a_referencia():
    gerador = np.random.default_rng(1)
    consumos = {f"R{i}": np.where(gerador.random(60) < 0.25, gerador.integers(1, 50, 60), 0).astype(float).tolist()
                for i in range(3)}
    consumos["R0"][:10] = [0.0] * 10                      # começa sem venda
    df = base_diaria(consumos)
    df = criar_medias_moveis(df, CHAVE_SERIE, (7,))
    df = criar_frequencias(df, CHAVE_SERIE, (7,))
    df = criar_recencia(df, CHAVE_SERIE)
    df = criar_tsb_sba(df, CHAVE_SERIE, alfa=0.2)
    for ref, valores in consumos.items():
        linhas = df["Referencia"] == ref
        esperado = referencia_intermitencia(valores, 0.2, 7)
        np.testing.assert_allclose(df.loc[linhas, "Freq_7"], esperado["Freq"], rtol=1e-6)
        np.testing.assert_allclose(df.loc[linhas, "DiasDesdeVenda"], esperado["DiasDesdeVenda"])
        np.testing.assert_allclose(df.loc[linhas, "DiasHistorico"], range(len(valores)))
        np.testing.assert_allclose(df.loc[linhas, "TSB"], esperado["TSB"], rtol=1e-5)
        np.testing.assert_allclose(df.loc[linhas, "SBA"], esperado["SBA"], rtol=1e-5)
        np.testing.assert_allclose(df.loc[linhas, "TamanhoMedio_7"],
                                   df.loc[linhas, "Media_7"] / df.loc[linhas, "Freq_7"].where(df.loc[linhas, "Freq_7"] > 0),
                                   rtol=1e-6)

    # sem vazamento: mudar o consumo do dia 40 não altera nenhuma feature até o dia 40
    alterado = base_diaria({**consumos, "R1": consumos["R1"][:40] + [999.0] + consumos["R1"][41:]})
    for funcao in (lambda d: criar_frequencias(d, CHAVE_SERIE, (7,)), lambda d: criar_recencia(d, CHAVE_SERIE),
                   lambda d: criar_tsb_sba(d, CHAVE_SERIE, alfa=0.2)):
        alterado = funcao(alterado)
    colunas = ["Freq_7", "DiasDesdeVenda", "TSB", "SBA"]
    ate_40 = (df["Referencia"] == "R1") & (df["Data"] <= df["Data"].min() + pd.Timedelta(days=40))
    pd.testing.assert_frame_equal(df.loc[ate_40, colunas], alterado.loc[ate_40, colunas])


def test_remover_sob_encomenda():
    df = pd.DataFrame({"Cliente": "K", "Referencia": ["A", "A", "B", "C", "D"], "IdBodega": "1",
                       "Descripcion": ["PSM", "PSM", "MTO", "DTO", None]})
    restante, linhas, series = remover_sob_encomenda(df, CHAVE_SERIE, ("TO",))
    assert restante["Referencia"].tolist() == ["A", "A", "D"] and (linhas, series) == (2, 2)
    assert restante.index.tolist() == [0, 1, 2]
    assert len(remover_sob_encomenda(df, CHAVE_SERIE, ())[0]) == 5


def test_hurdle_aprende_sinal():
    # séries intermitentes com escalas muito diferentes (1 a 1.000 por venda)
    gerador = np.random.default_rng(2)
    consumos = {}
    for i in range(60):
        tamanho, chance = 10 ** gerador.uniform(0, 3), gerador.uniform(0.05, 0.6)
        venda = gerador.random(500) < chance
        consumos[f"R{i}"] = np.where(venda, gerador.gamma(4.0, tamanho / 4.0, 500), 0).tolist()
    consumos["MORTA"] = [0.0] * 500
    df = base_diaria(consumos)
    df = criar_lags(df, CHAVE_SERIE, (1, 7))
    df = criar_medias_moveis(df, CHAVE_SERIE, (28, 90, 365))
    df = criar_frequencias(df, CHAVE_SERIE, (28, 90))
    df = criar_recencia(df, CHAVE_SERIE)
    df = criar_tsb_sba(df, CHAVE_SERIE)
    df = criar_alvo(df, CHAVE_SERIE, "Consumo_Limpo", 7)
    inicio_validacao, inicio_teste = datas_corte(df["Data"].max(), 60, 40)
    base = separar_bases(df, 7, inicio_validacao, inicio_teste)

    features = ("Lag_1", "Lag_7", "Media_28", "Media_90", "Freq_28", "Freq_90", "DiasDesdeVenda", "TSB", "SBA")
    escala = calcular_escala(df, ("Media_365", "Media_90"))
    assert (escala[df["Referencia"] == "MORTA"] == 0).all()
    parametros = {"min_data_in_leaf": 20, "num_leaves": 15}
    classificador, regressor = treinar_hurdle(df, base, features, PREFIXOS_ESCALAVEIS, escala,
                                              PARAMETROS_CLASSIFICADOR | parametros, PARAMETROS_REGRESSOR | parametros,
                                              300, 30)
    teste = (base == "teste").fillna(False).to_numpy()
    previsto, probabilidade = prever_hurdle(classificador, regressor, df, features, PREFIXOS_ESCALAVEIS, escala, teste)
    assert ((probabilidade >= 0) & (probabilidade <= 1)).all() and (previsto >= 0).all()
    assert (previsto[(df["Referencia"] == "MORTA").to_numpy()[teste]] == 0).all()

    real = df.loc[teste, "Alvo"].to_numpy()
    media_treino = df.loc[(base == "treino").fillna(False).to_numpy(), "Alvo"].mean()
    wape_modelo = np.abs(real - previsto).sum() / real.sum()
    wape_constante = np.abs(real - media_treino).sum() / real.sum()
    assert wape_modelo < 0.6 * wape_constante, (wape_modelo, wape_constante)


def test_metricas_feitas_a_mao():
    df = pd.DataFrame({
        "Cliente": "K", "Referencia": ["A", "A", "B", "B"], "IdBodega": "1",
        "Alvo": [2.0, 4.0, 0.0, 0.0],
        "Previsao": [3.0, 4.0, 1.0, 0.0],
        "Media_28": [1.0, 1.0, 0.0, 0.0],
    })
    por_serie = metricas_por_serie(df, CHAVE_SERIE, ("Previsao", "Media_28"))
    a = por_serie.set_index(["Modelo", "Referencia"])
    assert a.loc[("Previsao", "A"), "WAPE"] == 1 / 6
    assert a.loc[("Media_28", "A"), "WAPE"] == 4 / 6
    assert a.loc[("Previsao", "A"), "Vies"] == 7 / 6 - 1
    assert np.isnan(a.loc[("Previsao", "B"), "WAPE"])                 # série sem venda: só MAE
    assert a.loc[("Previsao", "B"), "MAE"] == 0.5
    assert a.loc[("Previsao", "A"), "RMSE"] == np.sqrt(0.5)
    assert a.loc[("Media_28", "A"), "RMSE"] == np.sqrt((1 + 9) / 2)

    resumo = comparar_modelos(por_serie, CHAVE_SERIE, "Previsao")
    assert resumo.loc["Previsao", "WAPE"] == 2 / 6
    assert resumo.loc["Media_28", "WAPE"] == 4 / 6
    assert resumo.loc["Previsao", "MAE"] == 0.5
    assert resumo.loc["Previsao", "RMSE"] == np.sqrt(2 / 4)
    assert resumo.loc["Media_28", "PrevisaoVence%"] == 100.0


def test_treino_aprende_sinal():
    # demanda intermitente com nível próprio por série: o modelo deve bater a média global do treino
    gerador = np.random.default_rng(0)
    consumos = {}
    for i in range(40):
        nivel = gerador.gamma(1.0, 5.0)
        venda = gerador.random(300) < 0.3
        consumos[f"R{i}"] = np.where(venda, gerador.poisson(nivel / 0.3), 0).astype(float).tolist()
    df = base_diaria(consumos)
    df = criar_lags(df, CHAVE_SERIE, (1, 7))
    df = criar_medias_moveis(df, CHAVE_SERIE, (7, 28))
    df = criar_alvo(df, CHAVE_SERIE, "Consumo_Limpo", 7)
    inicio_validacao, inicio_teste = datas_corte(df["Data"].max(), 60, 40)
    base = separar_bases(df, 7, inicio_validacao, inicio_teste)

    features = ("Lag_1", "Lag_7", "Media_7", "Media_28")
    modelo = treinar_lgbm(df, base, features, PARAMETROS_LGBM | {"min_data_in_leaf": 20}, 300, 30)
    teste = (base == "teste").fillna(False)
    real = df.loc[teste.to_numpy(), "Alvo"].to_numpy()
    previsto = prever(modelo, df, features, teste)
    media_treino = df.loc[(base == "treino").fillna(False).to_numpy(), "Alvo"].mean()
    wape_modelo = np.abs(real - previsto).sum() / real.sum()
    wape_constante = np.abs(real - media_treino).sum() / real.sum()
    assert (previsto >= 0).all()
    assert wape_modelo < 0.8 * wape_constante, (wape_modelo, wape_constante)


if __name__ == "__main__":
    test_alvo_a_frente()
    test_split_sem_sobreposicao()
    test_quartis_so_do_treino()
    test_adu_sobre_consumo_limpo()
    test_features_sem_picos()
    test_pico_vira_limite_superior()
    test_intermitencia_igual_a_referencia()
    test_remover_sob_encomenda()
    test_hurdle_aprende_sinal()
    test_metricas_feitas_a_mao()
    test_treino_aprende_sinal()
    print("Modelo: OK")
