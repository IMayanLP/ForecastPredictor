from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def calcular_escala(df: pd.DataFrame, colunas: tuple[str, ...]) -> np.ndarray:
    # 1ª coluna com valor > 0 (ex.: Media_365, senão Media_730); 0 = série sem venda nas janelas
    escala = np.zeros(len(df), dtype=np.float32)
    for coluna in reversed(colunas):
        valores = df[coluna].to_numpy(np.float32)
        escala = np.where(valores > 0, valores, escala)
    return escala


def matriz_normalizada(df: pd.DataFrame, features: tuple[str, ...], prefixos_escalaveis: tuple[str, ...],
                       escala: np.ndarray, linhas: np.ndarray) -> pd.DataFrame:
    # features de quantidade divididas pela escala da série: o modelo aprende "x vezes o normal da série"
    matriz = df.loc[linhas, list(features)].astype("float32")
    escala_linhas = escala[linhas]
    with np.errstate(divide="ignore", invalid="ignore"):
        for coluna in features:
            if coluna.startswith(prefixos_escalaveis):
                matriz[coluna] = matriz[coluna].to_numpy() / escala_linhas
    matriz["LogEscala"] = np.log1p(escala_linhas)
    return matriz


# Dataset montado uma vez e reaproveitado com outros parâmetros (busca): sem o pré-filtro, que fixa min_data_in_leaf
PARAMETROS_DATASET = {"feature_pre_filter": False, "verbose": -1}


def montar_matrizes(df: pd.DataFrame, base: pd.Series, features: tuple[str, ...], prefixos_escalaveis: tuple[str, ...],
                    escala: np.ndarray, alvo: str = "Alvo") -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    # treino e validação normalizados (matriz, alvo, escala); linhas sem escala ficam fora
    y = df[alvo].to_numpy(np.float32)
    matrizes = {}
    for nome in ("treino", "validacao"):
        linhas = (base == nome).fillna(False).to_numpy() & (escala > 0)
        matrizes[nome] = (matriz_normalizada(df, features, prefixos_escalaveis, escala, linhas), y[linhas], escala[linhas])
    return matrizes


def _par_de_datasets(matriz_treino, rotulo_treino, matriz_validacao, rotulo_validacao,
                     semente: int) -> tuple[lgb.Dataset, lgb.Dataset]:
    parametros = PARAMETROS_DATASET | {"seed": semente}
    dados_treino = lgb.Dataset(matriz_treino, rotulo_treino, params=parametros, free_raw_data=False)
    dados_validacao = lgb.Dataset(matriz_validacao, rotulo_validacao, reference=dados_treino, params=parametros,
                                  free_raw_data=False)
    return dados_treino, dados_validacao


def dados_classificador(matrizes: dict, semente: int) -> tuple[lgb.Dataset, lgb.Dataset]:
    # etapa 1: houve consumo em [t, t+H-1]?
    (x_treino, y_treino, _), (x_validacao, y_validacao, _) = matrizes["treino"], matrizes["validacao"]
    return _par_de_datasets(x_treino, (y_treino > 0).astype(np.float32),
                            x_validacao, (y_validacao > 0).astype(np.float32), semente)


def dados_regressor(matrizes: dict, semente: int) -> tuple[lgb.Dataset, lgb.Dataset]:
    # etapa 2: log(alvo / escala), só nas linhas com consumo
    (x_treino, y_treino, e_treino), (x_validacao, y_validacao, e_validacao) = matrizes["treino"], matrizes["validacao"]
    treino, validacao = y_treino > 0, y_validacao > 0
    return _par_de_datasets(x_treino[treino], np.log(y_treino[treino] / e_treino[treino]),
                            x_validacao[validacao], np.log(y_validacao[validacao] / e_validacao[validacao]), semente)


def treinar_etapa(dados: tuple[lgb.Dataset, lgb.Dataset], parametros: dict, rodadas: int, parada: int) -> lgb.Booster:
    dados_treino, dados_validacao = dados
    return lgb.train(PARAMETROS_DATASET | parametros, dados_treino, num_boost_round=rodadas,
                     valid_sets=[dados_validacao], valid_names=["validacao"],
                     callbacks=[lgb.early_stopping(parada, verbose=False)])


def treinar_classificador(matrizes: dict, parametros: dict, rodadas: int, parada: int) -> lgb.Booster:
    return treinar_etapa(dados_classificador(matrizes, parametros.get("seed", 0)), parametros, rodadas, parada)


def treinar_regressor(matrizes: dict, parametros: dict, rodadas: int, parada: int) -> lgb.Booster:
    return treinar_etapa(dados_regressor(matrizes, parametros.get("seed", 0)), parametros, rodadas, parada)


def treinar_hurdle(df: pd.DataFrame, base: pd.Series, features: tuple[str, ...], prefixos_escalaveis: tuple[str, ...],
                   escala: np.ndarray, parametros_classificador: dict, parametros_regressor: dict,
                   rodadas: int, parada: int, alvo: str = "Alvo") -> tuple[lgb.Booster, lgb.Booster]:
    # etapa 1: P(alvo > 0) | etapa 2: log(alvo / escala) só nas linhas com consumo; sem escala fica fora
    matrizes = montar_matrizes(df, base, features, prefixos_escalaveis, escala, alvo)
    return (treinar_classificador(matrizes, parametros_classificador, rodadas, parada),
            treinar_regressor(matrizes, parametros_regressor, rodadas, parada))


def prever_matriz(classificador: lgb.Booster, regressor: lgb.Booster, matriz: pd.DataFrame,
                  escala_linhas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # previsão = P(consumo) × tamanho relativo × escala; série sem escala → 0
    probabilidade = classificador.predict(matriz, num_iteration=classificador.best_iteration)
    tamanho = np.exp(regressor.predict(matriz, num_iteration=regressor.best_iteration)) * escala_linhas
    com_escala = escala_linhas > 0
    return np.where(com_escala, probabilidade * tamanho, 0.0), np.where(com_escala, probabilidade, 0.0)


def prever_hurdle(classificador: lgb.Booster, regressor: lgb.Booster, df: pd.DataFrame, features: tuple[str, ...],
                  prefixos_escalaveis: tuple[str, ...], escala: np.ndarray,
                  linhas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matriz = matriz_normalizada(df, features, prefixos_escalaveis, escala, linhas)
    return prever_matriz(classificador, regressor, matriz, escala[linhas])


def carregar_hurdle(pasta: Path) -> tuple[lgb.Booster, lgb.Booster]:
    # modelos gravados pelo main (só as árvores até a melhor rodada); prever com prever_hurdle / prever_matriz
    return (lgb.Booster(model_file=str(pasta / "hurdle_classificador.txt")),
            lgb.Booster(model_file=str(pasta / "hurdle_regressor.txt")))
