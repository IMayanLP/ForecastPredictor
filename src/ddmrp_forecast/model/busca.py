import hashlib
import itertools
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ddmrp_forecast.model.hurdle import dados_classificador, dados_regressor, prever_matriz, treinar_etapa
from ddmrp_forecast.model.metricas import comparar_modelos, medianas_por_segmento, metricas_por_serie, segmentar_abc
from ddmrp_forecast.model.split import inicio_da_base

# métrica de escolha (menor é melhor) de cada fase, sempre na validação
METRICA_CLASSIFICADOR = "LogLoss"
METRICA_REGRESSOR = "WAPETamanho"
METRICA_PRODUTO = "WAPEMediano"


def sortear_parametros(espaco: dict, semente: int, indice: int) -> dict:
    # tentativa 0 = parâmetros de settings; a tentativa i sai sempre igual (registro retomável)
    if indice == 0:
        return {}
    gerador = np.random.default_rng([semente, indice])
    sorteio = {}
    for nome, (tipo, *limites) in espaco.items():
        if tipo == "escolha":
            sorteio[nome] = limites[0][int(gerador.integers(len(limites[0])))]
            continue
        minimo, maximo = limites
        if tipo == "real":
            sorteio[nome] = float(gerador.uniform(minimo, maximo))
        elif tipo == "log_real":
            sorteio[nome] = float(np.exp(gerador.uniform(np.log(minimo), np.log(maximo))))
        elif tipo == "log_int":
            sorteio[nome] = int(round(np.exp(gerador.uniform(np.log(minimo), np.log(maximo)))))
        elif tipo == "zero_ou_log":
            # 1/3 das tentativas sem regularização, o resto em escala log
            valor = float(np.exp(gerador.uniform(np.log(minimo), np.log(maximo))))
            sorteio[nome] = 0.0 if gerador.random() < 1 / 3 else valor
        else:
            raise ValueError(f"Tipo de sorteio desconhecido: {tipo}")
    if sorteio.get("max_depth", -1) > 0 and "num_leaves" in sorteio:
        sorteio["num_leaves"] = min(sorteio["num_leaves"], 2 ** sorteio["max_depth"])
    return sorteio


def identificar_base(df: pd.DataFrame, base: pd.Series, configuracao: dict, alvo: str = "Alvo") -> str:
    # mesma base e mesma configuração → mesmo registro; qualquer mudança abre um registro novo
    resumo = {
        "linhas": len(df), "inicio": df["Data"].min(), "fim": df["Data"].max(),
        "alvo": round(float(df[alvo].sum()), 3), "bases": base.value_counts().sort_index().to_dict(), **configuracao,
    }
    return hashlib.sha1(json.dumps(resumo, sort_keys=True, default=str).encode()).hexdigest()[:10]


def ler_registro(caminho: Path) -> dict[str, dict]:
    if not caminho.exists():
        return {}
    registro = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        try:
            tentativa = json.loads(linha)
        except json.JSONDecodeError:
            continue  # linha cortada por uma interrupção no meio da gravação
        registro[tentativa["chave"]] = tentativa
    return registro


def gravar_tentativa(caminho: Path, tentativa: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(tentativa, ensure_ascii=False) + "\n")


def avaliar_classificador(modelo: lgb.Booster, matriz: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    real = y > 0
    probabilidade = np.clip(modelo.predict(matriz, num_iteration=modelo.best_iteration), 1e-7, 1 - 1e-7)
    return {
        "LogLoss": float(-np.mean(np.where(real, np.log(probabilidade), np.log(1 - probabilidade)))),
        "AUC": float(roc_auc_score(real, probabilidade)),
        "Brier": float(np.mean((probabilidade - real) ** 2)),
    }


def avaliar_regressor(modelo: lgb.Booster, matriz: pd.DataFrame, y: np.ndarray, escala: np.ndarray) -> dict[str, float]:
    # só linhas com consumo: tamanho previsto (exp × escala) contra o alvo
    previsto = np.exp(modelo.predict(matriz, num_iteration=modelo.best_iteration)) * escala
    razao = previsto / y
    return {
        "WAPETamanho": float(np.abs(previsto - y).sum() / y.sum()),
        "L1Log": float(np.mean(np.abs(np.log(razao)))),
        "Dentro25": float(np.mean(np.abs(razao - 1) <= 0.25)),
    }


def avaliar_produto(classificador: lgb.Booster, regressor: lgb.Booster, matrizes: dict, validacao: pd.DataFrame,
                    chave: tuple[str, ...], segmentos: pd.Series) -> dict[str, float]:
    # todas as linhas da validação, como no teste: sem escala → previsão 0
    matriz, _, escala = matrizes["validacao"]
    previsto = np.zeros(len(validacao))
    previsto[validacao["ComEscala"].to_numpy()] = prever_matriz(classificador, regressor, matriz, escala)[0]
    por_serie = metricas_por_serie(validacao.assign(Previsao=previsto), chave, ("Previsao",))
    total = comparar_modelos(por_serie, chave, "Previsao").loc["Previsao"]
    metricas = {"WAPEMediano": total["WAPEMedianoSerie"], "WAPE": total["WAPE"], "Vies": total["Vies"],
                "MAE": total["MAE"], "RMSE": total["RMSE"]}
    metricas |= {f"WAPEMediano{s}": v for s, v in medianas_por_segmento(por_serie, chave, segmentos).items()}
    return {nome: float(valor) for nome, valor in metricas.items()}


def ordenar(registro: dict[str, dict], fase: str, metrica: str, limite: int | None = None) -> list[dict]:
    tentativas = [t for t in registro.values() if t["fase"] == fase and (limite is None or t["indice"] < limite)]
    return sorted(tentativas, key=lambda t: (t["metricas"][metrica], t["chave"]))


def executar_busca(df: pd.DataFrame, base: pd.Series, matrizes: dict, chave: tuple[str, ...], coluna_volume: str,
                   escala: np.ndarray, pasta: Path, parametros_classificador: dict, parametros_regressor: dict,
                   espaco_classificador: dict, espaco_regressor: dict, tentativas_classificador: int,
                   tentativas_regressor: int, semente: int, melhores: int, sementes_extras: tuple[int, ...],
                   tolerancia_wape: float, rodadas: int, parada: int, alvo: str = "Alvo") -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / "tentativas.jsonl"
    registro = ler_registro(caminho)
    if registro:
        print(f"Registro retomado: {len(registro):,} tentativas já feitas em {caminho}")

    x_validacao, y_validacao, e_validacao = matrizes["validacao"]
    com_consumo = y_validacao > 0
    x_validacao_consumo = x_validacao[com_consumo]
    linhas_validacao = (base == "validacao").fillna(False).to_numpy()
    validacao = df.loc[linhas_validacao, [*chave, alvo]].assign(ComEscala=escala[linhas_validacao] > 0)
    # segmentos pelo volume antes da validação de cada cliente (só o treino é conhecido na escolha)
    antes_validacao = (df["Data"] < inicio_da_base(df, linhas_validacao)).to_numpy()
    segmentos = segmentar_abc(df, chave, coluna_volume, antes_validacao)

    etapas = {
        "classificador": dict(dados=dados_classificador, parametros=parametros_classificador, espaco=espaco_classificador,
                              tentativas=tentativas_classificador, metrica=METRICA_CLASSIFICADOR,
                              avaliar=lambda m: avaliar_classificador(m, x_validacao, y_validacao)),
        "regressor": dict(dados=dados_regressor, parametros=parametros_regressor, espaco=espaco_regressor,
                          tentativas=tentativas_regressor, metrica=METRICA_REGRESSOR,
                          avaliar=lambda m: avaliar_regressor(m, x_validacao_consumo, y_validacao[com_consumo],
                                                              e_validacao[com_consumo])),
    }
    # datasets montados uma vez por etapa (binning com a seed de settings) e reaproveitados em todas as tentativas
    datasets = {nome: e["dados"](matrizes, e["parametros"].get("seed", 0)) for nome, e in etapas.items()}
    modelos = {}

    def parametros_de(nome: str, indice: int, semente_modelo: int | None = None) -> dict:
        etapa = etapas[nome]
        parametros = etapa["parametros"] | sortear_parametros(etapa["espaco"], semente, indice)
        return parametros if semente_modelo is None else parametros | {"seed": semente_modelo}

    def modelo_de(nome: str, indice: int, semente_modelo: int | None = None) -> lgb.Booster:
        # treina sob demanda; guarda só o que ainda pode ser usado (melhores de cada etapa)
        if (nome, indice, semente_modelo) not in modelos:
            modelos[(nome, indice, semente_modelo)] = treinar_etapa(datasets[nome], parametros_de(nome, indice, semente_modelo),
                                                                     rodadas, parada)
        return modelos[(nome, indice, semente_modelo)]

    # fases A e B: cada etapa com sua métrica
    for nome, etapa in etapas.items():
        print(f"Busca — {nome}: {etapa['tentativas']} tentativas, escolha por {etapa['metrica']} na validação")
        for indice in range(etapa["tentativas"]):
            chave_tentativa = f"{nome}:{indice}"
            if chave_tentativa in registro:
                continue
            inicio = time.time()
            modelo = modelo_de(nome, indice)
            tentativa = {"chave": chave_tentativa, "fase": nome, "indice": indice, "parametros": parametros_de(nome, indice),
                         "melhor_rodada": modelo.best_iteration, "segundos": round(time.time() - inicio, 1),
                         "metricas": etapa["avaliar"](modelo)}
            gravar_tentativa(caminho, tentativa)
            registro[chave_tentativa] = tentativa
            print(f"  {chave_tentativa:<18} {etapa['metrica']} {tentativa['metricas'][etapa['metrica']]:.4f}  "
                  f"rodadas {modelo.best_iteration:>5,}  {tentativa['segundos']:>6.1f}s")
            manter = {t["indice"] for t in ordenar(registro, nome, etapa["metrica"], etapa["tentativas"])[:melhores]} | {0}
            for k in [k for k in modelos if k[0] == nome and k[1] not in manter]:
                del modelos[k]

    # fase C: produto dos melhores de cada etapa (+ parâmetros de settings), pelo WAPE mediano por série
    topo = {nome: sorted({t["indice"] for t in ordenar(registro, nome, e["metrica"], e["tentativas"])[:melhores]} | {0})
            for nome, e in etapas.items()}
    print(f"Busca — produto: {len(topo['classificador'])} × {len(topo['regressor'])} combinações, "
          f"escolha por {METRICA_PRODUTO} na validação")

    def avaliar_combinacao(fase: str, i: int, j: int, semente_modelo: int | None = None) -> dict:
        chave_tentativa = f"{fase}:{i}x{j}" + ("" if semente_modelo is None else f":{semente_modelo}")
        if chave_tentativa not in registro:
            inicio = time.time()
            metricas = avaliar_produto(modelo_de("classificador", i, semente_modelo),
                                       modelo_de("regressor", j, semente_modelo), matrizes, validacao, chave, segmentos)
            tentativa = {"chave": chave_tentativa, "fase": fase, "classificador": i, "regressor": j,
                         "semente": semente_modelo, "segundos": round(time.time() - inicio, 1), "metricas": metricas}
            gravar_tentativa(caminho, tentativa)
            registro[chave_tentativa] = tentativa
            print(f"  {chave_tentativa:<22} WAPE mediano {metricas['WAPEMediano']:.4f}  WAPE {metricas['WAPE']:.4f}  "
                  f"viés {metricas['Vies']:+.1%}")
        return registro[chave_tentativa]

    combinacoes = [avaliar_combinacao("produto", i, j)
                   for i, j in itertools.product(topo["classificador"], topo["regressor"])]
    # trava: o WAPE global não pode piorar mais que a tolerância em relação aos parâmetros de settings
    limite_wape = registro["produto:0x0"]["metricas"]["WAPE"] * (1 + tolerancia_wape)
    aceitas = sorted([c for c in combinacoes if c["metricas"]["WAPE"] <= limite_wape],
                     key=lambda c: (c["metricas"][METRICA_PRODUTO], c["chave"]))
    finalistas = {(c["classificador"], c["regressor"]) for c in aceitas[:melhores]} | {(0, 0)}

    # fase D: finalistas repetidos com outras sementes (bagging/feature_fraction); vence a média
    semente_base = etapas["classificador"]["parametros"].get("seed", 0)
    print(f"Busca — sementes: {len(finalistas)} finalistas × {1 + len(sementes_extras)} sementes")
    medias = []
    for i, j in sorted(finalistas):
        rodadas_finalista = [registro[f"produto:{i}x{j}"]["metricas"]]
        for extra in sementes_extras:
            rodadas_finalista.append(avaliar_combinacao("sementes", i, j, semente_base + extra)["metricas"])
        for k in [k for k in modelos if k[2] is not None]:
            del modelos[k]
        media = pd.DataFrame(rodadas_finalista).mean().to_dict()
        medias.append({"classificador": i, "regressor": j, **media})
    medias = pd.DataFrame(medias)
    medias["Aceita"] = medias["WAPE"] <= medias.loc[(medias["classificador"] == 0) & (medias["regressor"] == 0),
                                                     "WAPE"].iloc[0] * (1 + tolerancia_wape)
    medias = medias.sort_values(["Aceita", METRICA_PRODUTO], ascending=[False, True]).reset_index(drop=True)
    medias.to_csv(pasta / "finalistas.csv", sep=";", decimal=",", index=False)
    print("Finalistas (média das sementes, validação):")
    print(medias.to_string(float_format="{:,.4f}".format))

    vencedor = medias.iloc[0]
    i, j = int(vencedor["classificador"]), int(vencedor["regressor"])
    resultado = {
        "classificador": i, "regressor": j,
        "parametros_classificador": parametros_de("classificador", i),
        "parametros_regressor": parametros_de("regressor", j),
        "validacao": {k: float(v) for k, v in vencedor.items() if k not in ("classificador", "regressor", "Aceita")},
    }
    (pasta / "vencedor.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    resultado["modelos"] = {"vencedor": (modelo_de("classificador", i), modelo_de("regressor", j)),
                            "padrao": (modelo_de("classificador", 0), modelo_de("regressor", 0))}
    return resultado
