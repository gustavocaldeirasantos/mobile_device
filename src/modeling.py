"""Etapa 6 — Modelagem estatística e de machine learning.

Três modelos complementares:
    * Regressão Linear (OLS / statsmodels) — *explicativa*: o que dirige o gasto
      de bateria, com coeficientes interpretáveis e p-valores.
    * Regressão Logística multinomial — *preditiva/supervisionada*: classifica o
      perfil de uso (behavior_class) a partir das métricas de comportamento.
    * K-Means — *não supervisionada*: segmenta usuários e comparamos os clusters
      com as classes reais.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config

# Preditores do gasto de bateria (regressão) e da classe (classificação).
FEATURES: list[str] = [
    "app_usage_min",
    "screen_on_hours",
    "num_apps",
    "data_usage_mb",
    "age",
]


# --------------------------------------------------------------------------- #
# 1. Regressão Linear (explicativa)
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoRegressao:
    r2_treino: float
    r2_teste: float
    coeficientes: pd.DataFrame
    sumario: str = field(repr=False)


def treinar_regressao_linear(df: pd.DataFrame, alvo: str = "battery_drain_mah") -> ResultadoRegressao:
    """Ajusta uma Regressão Linear (OLS) para explicar o gasto de bateria.

    O statsmodels entrega p-valores e intervalos de confiança dos coeficientes,
    permitindo a leitura *inferencial* do modelo (não só a previsão).
    """
    preditores = [c for c in FEATURES if c != "age"] + ["age"]
    dados = df[preditores + [alvo]].dropna()
    X = dados[preditores]
    y = dados[alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)
    modelo = sm.OLS(y_train, X_train_c).fit()

    coefs = pd.DataFrame(
        {
            "coeficiente": modelo.params,
            "p_valor": modelo.pvalues,
            "ic_inferior": modelo.conf_int()[0],
            "ic_superior": modelo.conf_int()[1],
        }
    )
    return ResultadoRegressao(
        r2_treino=float(modelo.rsquared),
        r2_teste=float(r2_score(y_test, modelo.predict(X_test_c))),
        coeficientes=coefs,
        sumario=modelo.summary().as_text(),
    )


# --------------------------------------------------------------------------- #
# 2. Classificação (supervisionada)
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoClassificacao:
    acuracia: float
    relatorio: str = field(repr=False)
    matriz_confusao: np.ndarray = field(repr=False)
    importancias: pd.Series = field(repr=False)


def treinar_classificador(df: pd.DataFrame, alvo: str = "behavior_class") -> ResultadoClassificacao:
    """Treina uma Regressão Logística multinomial para prever o perfil de uso.

    As variáveis são padronizadas (a regularização da logística é sensível à
    escala). Avaliamos em conjunto de teste segregado, reportando acurácia,
    relatório por classe e matriz de confusão.
    """
    dados = df[FEATURES + [alvo]].dropna()
    X = dados[FEATURES]
    y = dados[alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    modelo = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    modelo.fit(scaler.transform(X_train), y_train)
    y_pred = modelo.predict(scaler.transform(X_test))

    # Importância = magnitude média dos coeficientes entre as classes.
    importancias = pd.Series(
        np.abs(modelo.coef_).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)

    return ResultadoClassificacao(
        acuracia=float((y_pred == y_test).mean()),
        relatorio=classification_report(y_test, y_pred, zero_division=0),
        matriz_confusao=confusion_matrix(y_test, y_pred),
        importancias=importancias.round(3),
    )


# --------------------------------------------------------------------------- #
# 3. Clusterização (não supervisionada)
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoClusterizacao:
    df_clusterizado: pd.DataFrame
    perfil_clusters: pd.DataFrame
    silhouette: float
    rand_ajustado: float


def segmentar_kmeans(df: pd.DataFrame, n_clusters: int = config.KMEANS_N_CLUSTERS) -> ResultadoClusterizacao:
    """Segmenta usuários com K-Means e compara com as classes reais.

    O *Adjusted Rand Index* mede o quanto os clusters não supervisionados
    concordam com a ``behavior_class`` verdadeira (1 = concordância perfeita).
    """
    dados = df[FEATURES + ["behavior_class"]].dropna().copy()
    X_scaled = StandardScaler().fit_transform(dados[FEATURES])

    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)
    dados["cluster"] = kmeans.fit_predict(X_scaled)

    perfil = (
        dados.groupby("cluster")
        .agg(
            n_usuarios=("age", "size"),
            app_usage_min=("app_usage_min", "mean"),
            screen_on_hours=("screen_on_hours", "mean"),
            num_apps=("num_apps", "mean"),
            data_usage_mb=("data_usage_mb", "mean"),
            classe_real_media=("behavior_class", "mean"),
        )
        .round(1)
    )
    return ResultadoClusterizacao(
        df_clusterizado=dados,
        perfil_clusters=perfil,
        silhouette=float(silhouette_score(X_scaled, dados["cluster"])),
        rand_ajustado=float(adjusted_rand_score(dados["behavior_class"], dados["cluster"])),
    )


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())

    print("\n=== REGRESSÃO LINEAR (gasto de bateria) ===")
    reg = treinar_regressao_linear(df)
    print(f"R² treino={reg.r2_treino:.3f} | teste={reg.r2_teste:.3f}")
    print(reg.coeficientes.round(4))

    print("\n=== CLASSIFICAÇÃO (perfil de uso) ===")
    clf = treinar_classificador(df)
    print(f"Acurácia: {clf.acuracia:.3f}")
    print(clf.relatorio)
    print("Importância das variáveis:\n", clf.importancias)

    print("\n=== K-MEANS ===")
    clu = segmentar_kmeans(df)
    print(f"Silhouette={clu.silhouette:.3f} | Rand ajustado={clu.rand_ajustado:.3f}")
    print(clu.perfil_clusters)
