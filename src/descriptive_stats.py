"""Etapa 3 — Estatística descritiva.

Quantifica o comportamento das variáveis numéricas: tendência central,
dispersão, forma da distribuição (assimetria/curtose) e normalidade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def tabela_descritiva(df: pd.DataFrame, colunas: list[str] | None = None) -> pd.DataFrame:
    """Monta uma tabela descritiva rica para as variáveis numéricas.

    Inclui medidas que o ``df.describe()`` não traz: moda, coeficiente de
    variação, assimetria (skewness), curtose e amplitude interquartil (IQR).
    """
    colunas = colunas or config.NUMERIC_COLS
    registros: list[dict[str, float]] = []

    for col in colunas:
        serie = df[col].dropna()
        media = float(serie.mean())
        desvio = float(serie.std())
        q1, q3 = np.percentile(serie, [25, 75])
        moda = serie.mode()
        registros.append(
            {
                "variavel": col,
                "n": int(serie.count()),
                "media": media,
                "mediana": float(serie.median()),
                "moda": float(moda.iloc[0]) if not moda.empty else np.nan,
                "desvio_padrao": desvio,
                "coef_variacao_%": (desvio / media * 100) if media else np.nan,
                "minimo": float(serie.min()),
                "q1": float(q1),
                "q3": float(q3),
                "maximo": float(serie.max()),
                "iqr": float(q3 - q1),
                # Assimetria: >0 cauda à direita, <0 cauda à esquerda.
                "assimetria": float(stats.skew(serie)),
                # Curtose (excesso): >0 caudas pesadas, <0 caudas leves.
                "curtose": float(stats.kurtosis(serie)),
            }
        )

    return pd.DataFrame(registros).set_index("variavel")


def teste_normalidade(df: pd.DataFrame, colunas: list[str] | None = None) -> pd.DataFrame:
    """Avalia a normalidade das distribuições (teste de D'Agostino-Pearson).

    H0: a amostra provém de uma distribuição normal.
    Se p-valor < alpha, rejeitamos H0 (distribuição não-normal).
    """
    colunas = colunas or config.NUMERIC_COLS
    registros: list[dict[str, object]] = []

    for col in colunas:
        serie = df[col].dropna()
        stat, p_valor = stats.normaltest(serie)
        registros.append(
            {
                "variavel": col,
                "estatistica": float(stat),
                "p_valor": float(p_valor),
                "normal_(alpha=%.2f)" % config.ALPHA: bool(p_valor > config.ALPHA),
            }
        )

    return pd.DataFrame(registros).set_index("variavel")


def resumo_categorico(df: pd.DataFrame, coluna: str, top: int = 10) -> pd.DataFrame:
    """Resume uma variável categórica: contagem e participação relativa."""
    contagem = df[coluna].value_counts().head(top)
    resumo = contagem.to_frame("frequencia")
    resumo["percentual_%"] = (contagem / len(df) * 100).round(2)
    return resumo


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print("\n=== TABELA DESCRITIVA ===")
    print(tabela_descritiva(df))
    print("\n=== TESTE DE NORMALIDADE ===")
    print(teste_normalidade(df))
    print("\n=== SISTEMA OPERACIONAL ===")
    print(resumo_categorico(df, "os"))
