"""Etapa 4 — Análise Exploratória Avançada (EDA).

* Detecção robusta de outliers por dois métodos (IQR e Z-score).
* Análise bivariada — relação das variáveis entre si e com a classe de
  comportamento (alvo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def detectar_outliers_iqr(
    df: pd.DataFrame, coluna: str, multiplicador: float = config.IQR_MULTIPLIER
) -> pd.Series:
    """Sinaliza outliers pelo método do Intervalo Interquartil (Tukey).

    Limites: [Q1 - k·IQR, Q3 + k·IQR]. Robusto por se basear em quantis.
    """
    serie = df[coluna]
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (serie < q1 - multiplicador * iqr) | (serie > q3 + multiplicador * iqr)


def detectar_outliers_zscore(
    df: pd.DataFrame, coluna: str, limite: float = config.ZSCORE_THRESHOLD
) -> pd.Series:
    """Sinaliza outliers pelo Z-score (|z| > limite). Pressupõe normalidade."""
    z = np.abs(stats.zscore(df[coluna], nan_policy="omit"))
    return pd.Series(z > limite, index=df.index)


def resumo_outliers(df: pd.DataFrame, colunas: list[str] | None = None) -> pd.DataFrame:
    """Compara a contagem de outliers detectados por IQR e por Z-score."""
    colunas = colunas or config.NUMERIC_COLS
    registros: list[dict[str, object]] = []
    for col in colunas:
        iqr_mask = detectar_outliers_iqr(df, col)
        z_mask = detectar_outliers_zscore(df, col)
        registros.append(
            {
                "variavel": col,
                "outliers_iqr": int(iqr_mask.sum()),
                "outliers_iqr_%": round(iqr_mask.mean() * 100, 2),
                "outliers_zscore": int(z_mask.sum()),
                "outliers_zscore_%": round(z_mask.mean() * 100, 2),
            }
        )
    return pd.DataFrame(registros).set_index("variavel")


def matriz_correlacao(
    df: pd.DataFrame, metodo: str = "pearson", colunas: list[str] | None = None
) -> pd.DataFrame:
    """Matriz de correlação (``pearson`` linear ou ``spearman`` monotônica)."""
    colunas = colunas or config.NUMERIC_COLS
    return df[colunas].corr(method=metodo)


def analise_bivariada_alvo(df: pd.DataFrame, alvo: str = "behavior_class") -> pd.DataFrame:
    """Correlação (Pearson e Spearman) de cada métrica numérica com o alvo.

    Mostra quais comportamentos de uso mais se associam à classe do usuário.
    """
    registros: list[dict[str, object]] = []
    for col in config.NUMERIC_COLS:
        sub = df[[alvo, col]].dropna()
        r_p, p_p = stats.pearsonr(sub[alvo], sub[col])
        r_s, p_s = stats.spearmanr(sub[alvo], sub[col])
        registros.append(
            {
                "variavel": col,
                "pearson_r": round(float(r_p), 4),
                "pearson_p": float(p_p),
                "spearman_r": round(float(r_s), 4),
                "spearman_p": float(p_s),
            }
        )
    return (
        pd.DataFrame(registros)
        .set_index("variavel")
        .sort_values("spearman_r", ascending=False)
    )


def perfil_por_classe(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil médio de uso para cada classe de comportamento."""
    return (
        df.groupby("behavior_label")[config.NUMERIC_COLS]
        .mean()
        .round(1)
    )


def perfil_por_so(df: pd.DataFrame) -> pd.DataFrame:
    """Comparativo de uso médio entre Android e iOS."""
    return df.groupby("os")[config.NUMERIC_COLS].mean().round(1)


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())
    pd.set_option("display.width", 180)
    print("\n=== OUTLIERS ===")
    print(resumo_outliers(df))
    print("\n=== CORRELAÇÃO (PEARSON) ===")
    print(matriz_correlacao(df).round(3))
    print("\n=== BIVARIADA COM A CLASSE ===")
    print(analise_bivariada_alvo(df))
    print("\n=== PERFIL POR CLASSE ===")
    print(perfil_por_classe(df))
