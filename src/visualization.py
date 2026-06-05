"""Módulo de visualização — funções de plotagem reutilizáveis.

Separar a visualização da lógica analítica mantém os módulos de estatística
"puros" e concentra aqui a dependência de Matplotlib/Seaborn.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid", palette="mako")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"


def _salvar(fig: plt.Figure, nome: str | None) -> None:
    if nome:
        config.ensure_directories()
        caminho = config.FIGURES_DIR / nome
        fig.savefig(caminho, bbox_inches="tight")
        print(f"[visualization] Figura salva em {caminho}")


def plot_distribuicoes(df: pd.DataFrame, salvar_como: str | None = None) -> plt.Figure:
    """Painel de histogramas das variáveis numéricas."""
    cols = config.NUMERIC_COLS
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#3b6978")
        ax.set_title(col)
    fig.suptitle("Distribuição das métricas de uso", fontweight="bold", y=1.02)
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_boxplots_outliers(df: pd.DataFrame, salvar_como: str | None = None) -> plt.Figure:
    """Boxplots padronizados (z-score) para visualizar outliers lado a lado."""
    padronizado = df[config.NUMERIC_COLS].apply(lambda s: (s - s.mean()) / s.std())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=padronizado, ax=ax, orient="h")
    ax.set_title("Boxplots padronizados (z-score) — detecção de outliers")
    ax.set_xlabel("Desvios-padrão em relação à média")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_matriz_correlacao(matriz: pd.DataFrame, titulo: str, salvar_como: str | None = None) -> plt.Figure:
    """Heatmap de uma matriz de correlação."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mascara = np.triu(np.ones_like(matriz, dtype=bool))
    sns.heatmap(matriz, mask=mascara, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title(titulo)
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_uso_por_classe(df: pd.DataFrame, coluna: str = "app_usage_min", salvar_como: str | None = None) -> plt.Figure:
    """Boxplot de uma métrica por classe de comportamento."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordem = sorted(df["behavior_label"].dropna().unique())
    sns.boxplot(data=df, x="behavior_label", y=coluna, order=ordem, ax=ax)
    ax.set_title(f"{coluna} por classe de comportamento")
    ax.set_xlabel("Classe de comportamento")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_android_vs_ios(df: pd.DataFrame, coluna: str = "battery_drain_mah", salvar_como: str | None = None) -> plt.Figure:
    """Comparação da distribuição de uma métrica entre Android e iOS."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x="os", y=coluna, ax=ax, inner="quartile")
    ax.set_title(f"{coluna}: Android vs iOS")
    ax.set_xlabel("Sistema operacional")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_matriz_confusao(matriz: np.ndarray, salvar_como: str | None = None) -> plt.Figure:
    """Heatmap da matriz de confusão do classificador."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    labels = [config.BEHAVIOR_LABELS[i].split(" - ")[0] for i in sorted(config.BEHAVIOR_LABELS)]
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Matriz de confusão — previsão do perfil de uso")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_clusters(df_clusterizado: pd.DataFrame, salvar_como: str | None = None) -> plt.Figure:
    """Dispersão tempo de app × dados, colorida pelos clusters do K-Means."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_clusterizado, x="app_usage_min", y="data_usage_mb",
                    hue="cluster", palette="mako", alpha=0.7, s=35, ax=ax)
    ax.set_title("Segmentos de usuários identificados por K-Means")
    ax.set_xlabel("Tempo de uso de apps (min/dia)")
    ax.set_ylabel("Uso de dados (MB/dia)")
    ax.legend(title="Cluster")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.eda_advanced import matriz_correlacao
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())
    plot_distribuicoes(df, salvar_como="dist.png")
    plot_matriz_correlacao(matriz_correlacao(df), "Correlação (Pearson)", salvar_como="corr.png")
    print("Figuras de exemplo geradas.")
