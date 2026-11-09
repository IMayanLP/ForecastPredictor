import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from ddmrp_forecast.features.janelas import agrupar_series


def marcar_acerto(alvo: np.ndarray, previsao: np.ndarray, escala: np.ndarray, tolerancia: float) -> np.ndarray:
    # relativo ao maior entre o alvo e o normal da série: vale com alvo 0 e não depende da unidade
    return np.abs(previsao - alvo) <= tolerancia * np.maximum(alvo, escala)


def preparar_sinais(linhas: pd.DataFrame, previsao: np.ndarray, probabilidade: np.ndarray,
                    escala: np.ndarray) -> pd.DataFrame:
    linhas["Previsao"], linhas["Probabilidade"], linhas["Escala"] = previsao, probabilidade, escala
    linhas["PrevSobreEscala"] = np.divide(previsao, escala, out=np.full(len(linhas), np.nan), where=escala > 0)
    return linhas


def dobras_por_serie(linhas: pd.DataFrame, chave: tuple[str, ...], dobras: int) -> np.ndarray:
    # origens seguidas da mesma série se sobrepõem e não podem ficar dos dois lados
    return agrupar_series(linhas, chave).ngroup().to_numpy() % dobras


def ajustar_confiabilidade(validacao: pd.DataFrame, chave: tuple[str, ...], features: tuple[str, ...], tolerancia: float,
                           parametros: dict, dobras: int, rodadas: int, parada: int) -> dict:
    # linhas sem escala (previsão 0) não passam pelo modelo: recebem a taxa de acerto da validação
    escala = validacao["Escala"].to_numpy(float)
    acerto = marcar_acerto(validacao["Alvo"].to_numpy(float), validacao["Previsao"].to_numpy(float), escala, tolerancia)
    com_escala = escala > 0
    taxa_sem_escala = float(acerto[~com_escala].mean()) if (~com_escala).any() else 0.0

    linhas = validacao[com_escala]
    matriz = linhas[list(features)].astype("float32")
    rotulo = acerto[com_escala].astype(np.float32)
    codigos = dobras_por_serie(linhas, chave, dobras)
    fora_da_dobra = np.zeros(len(linhas))
    melhores = []
    for dobra in range(dobras):
        controle = codigos == dobra
        dados = lgb.Dataset(matriz[~controle], rotulo[~controle], free_raw_data=False)
        modelo = lgb.train(parametros, dados, num_boost_round=rodadas,
                           valid_sets=[lgb.Dataset(matriz[controle], rotulo[controle], reference=dados)],
                           callbacks=[lgb.early_stopping(parada, verbose=False)])
        fora_da_dobra[controle] = modelo.predict(matriz[controle], num_iteration=modelo.best_iteration)
        melhores.append(modelo.best_iteration)

    final = lgb.train(parametros, lgb.Dataset(matriz, rotulo), num_boost_round=max(int(np.mean(melhores)), 1))
    isotonica = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(fora_da_dobra, rotulo)
    return {
        "modelo": final,
        "calibracao_x": isotonica.X_thresholds_.tolist(),
        "calibracao_y": isotonica.y_thresholds_.tolist(),
        "taxa_sem_escala": taxa_sem_escala,
        "features": list(features),
        "tolerancia": tolerancia,
        "rodadas_por_dobra": melhores,
        "auc_fora_da_dobra": float(roc_auc_score(rotulo, fora_da_dobra)),
    }


def aplicar_confiabilidade(linhas: pd.DataFrame, ajuste: dict) -> np.ndarray:
    # não usa o alvo: serve também para origens recentes, sem janela fechada
    com_escala = linhas["Escala"].to_numpy(float) > 0
    confiabilidade = np.full(len(linhas), ajuste["taxa_sem_escala"], dtype=float)
    if com_escala.any():
        bruto = ajuste["modelo"].predict(linhas.loc[com_escala, ajuste["features"]].astype("float32"))
        confiabilidade[com_escala] = np.interp(bruto, ajuste["calibracao_x"], ajuste["calibracao_y"])
    return confiabilidade


def classificar(confiabilidade: np.ndarray, cortes: tuple[float, ...], rotulos: tuple[str, ...]) -> pd.Categorical:
    return pd.Categorical.from_codes(np.digitize(confiabilidade, cortes), categories=list(rotulos), ordered=True)


def tipo_previsao(probabilidade: np.ndarray, corte: float) -> np.ndarray:
    return np.where(probabilidade >= corte, "Com consumo", "Sem consumo")


def gravar_confiabilidade(ajuste: dict, pasta: Path) -> None:
    ajuste["modelo"].save_model(pasta / "confiabilidade_modelo.txt")
    dados = {nome: valor for nome, valor in ajuste.items() if nome != "modelo"}
    (pasta / "confiabilidade_calibracao.json").write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_confiabilidade(pasta: Path) -> dict:
    dados = json.loads((pasta / "confiabilidade_calibracao.json").read_text(encoding="utf-8"))
    return dados | {"modelo": lgb.Booster(model_file=str(pasta / "confiabilidade_modelo.txt"))}


def tabela_por_classe(linhas: pd.DataFrame, chave: tuple[str, ...]) -> pd.DataFrame:
    # precisa de Acerto, Confiabilidade, ClasseConfiabilidade, Alvo e Previsao; classe vazia continua na tabela
    linhas = linhas.assign(ErroAbs=np.abs(linhas["Previsao"] - linhas["Alvo"]), AlvoZero=linhas["Alvo"] == 0)
    tabela = linhas.groupby("ClasseConfiabilidade", observed=False).agg(
        Linhas=("Acerto", "size"), Estimada=("Confiabilidade", "mean"), Acerto=("Acerto", "mean"),
        AlvoZero=("AlvoZero", "mean"), SomaErro=("ErroAbs", "sum"), SomaAlvo=("Alvo", "sum"),
    )
    tabela["Distribuicao"] = tabela["Linhas"] / max(len(linhas), 1)
    tabela["WAPE"] = (tabela["SomaErro"] / tabela["SomaAlvo"]).where(tabela["SomaAlvo"] > 0)
    por_serie = linhas.groupby(["ClasseConfiabilidade", *chave], observed=True).agg(
        SomaErro=("ErroAbs", "sum"), SomaAlvo=("Alvo", "sum"))
    wape_serie = (por_serie["SomaErro"] / por_serie["SomaAlvo"]).where(por_serie["SomaAlvo"] > 0)
    tabela["Series"] = wape_serie.groupby("ClasseConfiabilidade", observed=False).size()
    tabela["WAPEMediano"] = wape_serie.groupby("ClasseConfiabilidade", observed=False).median()
    return tabela[["Linhas", "Distribuicao", "Series", "Estimada", "Acerto", "AlvoZero", "WAPE", "WAPEMediano"]]


def relatorio_por_recorte(teste: pd.DataFrame, chave: tuple[str, ...]) -> pd.DataFrame:
    # sem escala à parte: é previsão 0 que acerta quase sempre e inflaria as demais tabelas
    com_escala = teste["Escala"] > 0
    recortes = {"Com escala": com_escala, "Todas": pd.Series(True, index=teste.index), "Sem escala": ~com_escala}
    for segmento in ("A", "B", "C"):
        recortes[f"Segmento {segmento}"] = com_escala & (teste["Segmento"] == segmento)
    for tipo in ("Com consumo", "Sem consumo"):
        recortes[tipo] = com_escala & (teste["TipoPrevisao"] == tipo)
    clientes = teste["Cliente"].unique()
    if len(clientes) > 1:
        for cliente in clientes:
            recortes[f"Cliente {cliente}"] = com_escala & (teste["Cliente"] == cliente)
    return pd.concat({nome: tabela_por_classe(teste[filtro], chave) for nome, filtro in recortes.items()},
                     names=["Recorte"])


def eh_monotona(tabela: pd.DataFrame, linhas_minimas: int) -> bool:
    acerto = tabela.loc[tabela["Linhas"] >= linhas_minimas, "Acerto"].to_numpy()
    return bool((np.diff(acerto) > 0).all())


def bootstrap_por_serie(linhas: pd.DataFrame, chave: tuple[str, ...], repeticoes: int, semente: int,
                        linhas_minimas: int) -> tuple[pd.DataFrame, float]:
    # sorteia séries inteiras (origens da mesma série não são independentes): IC 90% do acerto e % de sorteios monótonos
    classes = linhas["ClasseConfiabilidade"].cat.categories
    contagem = linhas.groupby([*chave, "ClasseConfiabilidade"], observed=True)["Acerto"].agg(["sum", "size"]).unstack(
        "ClasseConfiabilidade", fill_value=0)
    acertos = contagem["sum"].reindex(columns=classes, fill_value=0).to_numpy(float)
    tamanhos = contagem["size"].reindex(columns=classes, fill_value=0).to_numpy(float)
    avaliadas = tamanhos.sum(axis=0) >= linhas_minimas
    gerador = np.random.default_rng(semente)
    taxas = np.full((repeticoes, len(classes)), np.nan)
    for repeticao in range(repeticoes):
        sorteio = gerador.integers(0, len(acertos), len(acertos))
        total = tamanhos[sorteio].sum(axis=0)
        taxas[repeticao] = np.divide(acertos[sorteio].sum(axis=0), total, out=np.full(len(classes), np.nan), where=total > 0)
    intervalo = pd.DataFrame(np.nanpercentile(taxas[:, avaliadas], [5, 95], axis=0).T, columns=["Acerto_p5", "Acerto_p95"],
                             index=classes[avaliadas])
    # classe que sumiu no sorteio (NaN) não conta contra: compara só as presentes
    monotonos = [bool((np.diff(linha[~np.isnan(linha)]) > 0).all()) for linha in taxas[:, avaliadas]]
    return intervalo, float(np.mean(monotonos))


def avaliar_discriminacao(acerto: np.ndarray, probabilidade: np.ndarray, segmentos: np.ndarray) -> dict[str, float]:
    medidas = {"AUC": roc_auc_score(acerto, probabilidade), "Brier": brier_score_loss(acerto, probabilidade)}
    for segmento in ("A", "B", "C"):
        filtro = segmentos == segmento
        tem_as_duas = filtro.any() and acerto[filtro].min() != acerto[filtro].max()
        medidas[f"AUC_{segmento}"] = roc_auc_score(acerto[filtro], probabilidade[filtro]) if tem_as_duas else np.nan
    return medidas


def _faixas_da_regra(linhas: pd.DataFrame, colunas: tuple[str, ...], faixas: dict) -> pd.Series:
    partes = pd.Series("", index=linhas.index)
    for coluna in colunas:
        if coluna in faixas:
            cortes, valor_nulo = faixas[coluna]
            valores = linhas[coluna].fillna(valor_nulo).to_numpy(float)
            partes = partes + np.digitize(valores, cortes, right=True).astype(str) + "|"
        else:
            partes = partes + linhas[coluna].astype(str) + "|"
    return partes


def taxa_da_regra(validacao: pd.DataFrame, teste: pd.DataFrame, colunas: tuple[str, ...], faixas: dict) -> np.ndarray:
    # % = taxa de acerto da faixa na validação; faixa que não existia nela recebe a taxa geral
    taxas = validacao["Acerto"].astype(float).groupby(_faixas_da_regra(validacao, colunas, faixas)).mean()
    return _faixas_da_regra(teste, colunas, faixas).map(taxas).fillna(validacao["Acerto"].mean()).to_numpy(float)


def comparar_com_regras(validacao: pd.DataFrame, teste: pd.DataFrame, regras: dict, faixas: dict,
                        cortes: tuple[float, ...], rotulos: tuple[str, ...]) -> pd.DataFrame:
    # mesmas linhas para todos; cada método com a sua %, as mesmas faixas e o acerto real em cada classe
    concorrentes = {"Modelo": teste["Confiabilidade"].to_numpy(float)}
    concorrentes |= {nome: taxa_da_regra(validacao, teste, colunas, faixas) for nome, colunas in regras.items()}
    acerto = teste["Acerto"].to_numpy(float)
    segmentos = teste["Segmento"].to_numpy()
    tabela = {}
    for nome, estimada in concorrentes.items():
        classes = classificar(estimada, cortes, rotulos)
        taxas = pd.Series(acerto).groupby(classes, observed=False).mean()
        tabela[nome] = avaliar_discriminacao(acerto, estimada, segmentos) | taxas.to_dict()
    return pd.DataFrame(tabela).T
