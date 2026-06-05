"""Etapa 5 — Estatística inferencial.

Testa hipóteses sobre a população e quantifica a incerteza das estimativas.
    * Teste t de Welch — diferença de médias entre dois grupos (Android vs iOS).
    * ANOVA — diferença de médias entre as classes de comportamento.
    * Qui-quadrado — associação entre variáveis categóricas.
    * Intervalos de confiança — paramétrico (t) e bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src import config


@dataclass
class ResultadoTeste:
    """Estrutura tipada para o resultado de um teste de hipótese."""

    nome: str
    estatistica: float
    p_valor: float
    significativo: bool
    interpretacao: str

    def __str__(self) -> str:
        return (
            f"[{self.nome}] estatística={self.estatistica:.4f}, "
            f"p-valor={self.p_valor:.4g} -> {self.interpretacao}"
        )


def teste_t_android_vs_ios(df: pd.DataFrame, coluna: str = "battery_drain_mah") -> ResultadoTeste:
    """Teste t de Welch: Android e iOS diferem em ``coluna``?

    Usamos o t de Welch (``equal_var=False``) por não assumir variâncias iguais,
    a escolha segura para grupos de tamanhos diferentes (554 Android vs 146 iOS).

    H0: as médias dos dois sistemas operacionais são iguais.
    """
    android = df.loc[df["os"] == "Android", coluna].dropna()
    ios = df.loc[df["os"] == "iOS", coluna].dropna()
    stat, p = stats.ttest_ind(android, ios, equal_var=False)
    significativo = bool(p < config.ALPHA)
    interp = (
        f"Diferença significativa (Android={android.mean():,.1f} vs "
        f"iOS={ios.mean():,.1f})."
        if significativo
        else f"Sem diferença significativa (Android={android.mean():,.1f} vs "
        f"iOS={ios.mean():,.1f})."
    )
    return ResultadoTeste(f"Teste t Android vs iOS ({coluna})", float(stat), float(p), significativo, interp)


def teste_t_genero(df: pd.DataFrame, coluna: str = "app_usage_min") -> ResultadoTeste:
    """Teste t de Welch: homens e mulheres diferem em ``coluna``?"""
    h = df.loc[df["gender"] == "Male", coluna].dropna()
    m = df.loc[df["gender"] == "Female", coluna].dropna()
    stat, p = stats.ttest_ind(h, m, equal_var=False)
    significativo = bool(p < config.ALPHA)
    interp = (
        f"Diferença significativa entre gêneros (M={h.mean():,.1f} vs F={m.mean():,.1f})."
        if significativo
        else f"Sem diferença significativa entre gêneros (M={h.mean():,.1f} vs F={m.mean():,.1f})."
    )
    return ResultadoTeste(f"Teste t por gênero ({coluna})", float(stat), float(p), significativo, interp)


def anova_classes(df: pd.DataFrame, coluna: str = "app_usage_min") -> ResultadoTeste:
    """ANOVA de uma via: ``coluna`` difere entre as classes de comportamento?

    H0: todas as classes têm a mesma média populacional de ``coluna``.
    """
    grupos = [g[coluna].dropna().values for _, g in df.groupby("behavior_class")]
    stat, p = stats.f_oneway(*grupos)
    significativo = bool(p < config.ALPHA)
    interp = (
        f"Ao menos uma classe difere em '{coluna}' (a métrica separa os perfis)."
        if significativo
        else f"Não há diferença de '{coluna}' entre as classes."
    )
    return ResultadoTeste(f"ANOVA por classe ({coluna})", float(stat), float(p), significativo, interp)


def qui_quadrado(df: pd.DataFrame, col_a: str = "os", col_b: str = "behavior_class") -> ResultadoTeste:
    """Teste qui-quadrado de independência entre duas variáveis categóricas.

    H0: as variáveis são independentes (sem associação).
    """
    tabela = pd.crosstab(df[col_a], df[col_b])
    stat, p, _, _ = stats.chi2_contingency(tabela)
    significativo = bool(p < config.ALPHA)
    interp = (
        f"'{col_a}' e '{col_b}' são associados (dependentes)."
        if significativo
        else f"'{col_a}' e '{col_b}' são independentes."
    )
    return ResultadoTeste(f"Qui-quadrado ({col_a} x {col_b})", float(stat), float(p), significativo, interp)


def intervalo_confianca_t(serie: pd.Series, confianca: float = 1 - config.ALPHA) -> tuple[float, float]:
    """Intervalo de confiança paramétrico para a média (t de Student)."""
    dados = serie.dropna()
    n = len(dados)
    margem = stats.sem(dados) * stats.t.ppf((1 + confianca) / 2, df=n - 1)
    return float(dados.mean() - margem), float(dados.mean() + margem)


def intervalo_confianca_bootstrap(
    serie: pd.Series, confianca: float = 1 - config.ALPHA, n_reamostragens: int = 5_000
) -> tuple[float, float]:
    """Intervalo de confiança via bootstrap (percentil). Não exige normalidade."""
    dados = serie.dropna().to_numpy()
    rng = np.random.default_rng(config.RANDOM_STATE)
    amostras = rng.choice(dados, size=(n_reamostragens, len(dados)), replace=True)
    medias = amostras.mean(axis=1)
    alpha = 1 - confianca
    low, high = np.percentile(medias, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def tabela_intervalos_confianca(df: pd.DataFrame, colunas: list[str] | None = None) -> pd.DataFrame:
    """Compara IC paramétrico (t) e bootstrap para a média de cada variável."""
    colunas = colunas or config.NUMERIC_COLS
    registros: list[dict[str, object]] = []
    for col in colunas:
        t_low, t_high = intervalo_confianca_t(df[col])
        b_low, b_high = intervalo_confianca_bootstrap(df[col])
        registros.append(
            {
                "variavel": col,
                "media": float(df[col].mean()),
                "ic95_t_inferior": t_low,
                "ic95_t_superior": t_high,
                "ic95_boot_inferior": b_low,
                "ic95_boot_superior": b_high,
            }
        )
    return pd.DataFrame(registros).set_index("variavel")


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())
    print(teste_t_android_vs_ios(df))
    print(teste_t_genero(df))
    print(anova_classes(df))
    print(qui_quadrado(df))
    print("\n=== INTERVALOS DE CONFIANÇA (95%) ===")
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(tabela_intervalos_confianca(df))
