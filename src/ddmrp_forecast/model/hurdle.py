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


def treinar_hurdle(df: pd.DataFrame, base: pd.Series, features: tuple[str, ...], prefixos_escalaveis: tuple[str, ...],
                   escala: np.ndarray, parametros_classificador: dict, parametros_regressor: dict,
                   rodadas: int, parada: int, alvo: str = "Alvo") -> tuple[lgb.Booster, lgb.Booster]:
    # etapa 1: P(alvo > 0) | etapa 2: log(alvo / escala) só nas linhas com consumo; sem escala fica fora
    com_escala = escala > 0
    y = df[alvo].to_numpy(np.float32)
    treino = (base == "treino").fillna(False).to_numpy() & com_escala
    validacao = (base == "validacao").fillna(False).to_numpy() & com_escala

    def ajustar(parametros, linhas_treino, linhas_validacao, transformar):
        dados_treino = lgb.Dataset(matriz_normalizada(df, features, prefixos_escalaveis, escala, linhas_treino),
                                   transformar(linhas_treino))
        dados_validacao = lgb.Dataset(matriz_normalizada(df, features, prefixos_escalaveis, escala, linhas_validacao),
                                      transformar(linhas_validacao), reference=dados_treino)
        return lgb.train(parametros, dados_treino, num_boost_round=rodadas, valid_sets=[dados_validacao],
                         valid_names=["validacao"], callbacks=[lgb.early_stopping(parada, verbose=False)])

    classificador = ajustar(parametros_classificador, treino, validacao, lambda linhas: (y[linhas] > 0).astype(np.float32))
    com_consumo = y > 0
    regressor = ajustar(parametros_regressor, treino & com_consumo, validacao & com_consumo,
                        lambda linhas: np.log(y[linhas] / escala[linhas]))
    return classificador, regressor


def prever_hurdle(classificador: lgb.Booster, regressor: lgb.Booster, df: pd.DataFrame, features: tuple[str, ...],
                  prefixos_escalaveis: tuple[str, ...], escala: np.ndarray,
                  linhas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # previsão = P(consumo) × tamanho relativo × escala; série sem escala → 0
    matriz = matriz_normalizada(df, features, prefixos_escalaveis, escala, linhas)
    probabilidade = classificador.predict(matriz, num_iteration=classificador.best_iteration)
    tamanho = np.exp(regressor.predict(matriz, num_iteration=regressor.best_iteration)) * escala[linhas]
    com_escala = escala[linhas] > 0
    return np.where(com_escala, probabilidade * tamanho, 0.0), np.where(com_escala, probabilidade, 0.0)
