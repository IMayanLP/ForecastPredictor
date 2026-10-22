import lightgbm as lgb
import numpy as np
import pandas as pd


def matriz(df: pd.DataFrame, features: tuple[str, ...], linhas: pd.Series) -> pd.DataFrame:
    return df.loc[linhas.to_numpy(), list(features)].astype("float32")


def treinar_lgbm(df: pd.DataFrame, base: pd.Series, features: tuple[str, ...], parametros: dict,
                 rodadas: int, parada: int, alvo: str = "Alvo") -> lgb.Booster:
    treino = (base == "treino").fillna(False)
    validacao = (base == "validacao").fillna(False)
    dados_treino = lgb.Dataset(matriz(df, features, treino), df.loc[treino.to_numpy(), alvo].to_numpy(np.float32),
                               free_raw_data=True)
    dados_validacao = lgb.Dataset(matriz(df, features, validacao),
                                  df.loc[validacao.to_numpy(), alvo].to_numpy(np.float32), reference=dados_treino)
    return lgb.train(
        parametros, dados_treino, num_boost_round=rodadas,
        valid_sets=[dados_treino, dados_validacao], valid_names=["treino", "validacao"],
        callbacks=[lgb.early_stopping(parada, verbose=False), lgb.log_evaluation(0)],
    )


def prever(modelo: lgb.Booster, df: pd.DataFrame, features: tuple[str, ...], linhas: pd.Series) -> np.ndarray:
    return modelo.predict(matriz(df, features, linhas), num_iteration=modelo.best_iteration)


def importancia(modelo: lgb.Booster) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "Feature": modelo.feature_name(),
            "Ganho": modelo.feature_importance("gain"),
            "Divisoes": modelo.feature_importance("split"),
        })
        .assign(GanhoPct=lambda t: t["Ganho"] / t["Ganho"].sum() * 100)
        .sort_values("Ganho", ascending=False, ignore_index=True)
    )
