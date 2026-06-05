"""Etapa 2 — Limpeza, padronização e engenharia de atributos.

O dataset original é sintético e bem comportado (sem nulos), então o foco aqui
é: padronizar nomes de colunas, garantir tipos corretos, rotular o alvo e criar
variáveis derivadas com valor analítico.
"""
from __future__ import annotations

import pandas as pd

from src import config

# Mapeamento dos nomes originais (com unidades) para nomes limpos em snake_case.
COLUMN_RENAME: dict[str, str] = {
    "User ID": "user_id",
    "Device Model": "device_model",
    "Operating System": "os",
    "App Usage Time (min/day)": "app_usage_min",
    "Screen On Time (hours/day)": "screen_on_hours",
    "Battery Drain (mAh/day)": "battery_drain_mah",
    "Number of Apps Installed": "num_apps",
    "Data Usage (MB/day)": "data_usage_mb",
    "Age": "age",
    "Gender": "gender",
    "User Behavior Class": "behavior_class",
}


def _padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas para snake_case sem unidades no nome."""
    return df.rename(columns=COLUMN_RENAME)


def _criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas de alto valor analítico."""
    df = df.copy()

    # Consumo de dados por aplicativo instalado.
    df["data_por_app"] = df["data_usage_mb"] / df["num_apps"]

    # Bateria consumida por hora de tela ligada (eficiência energética do uso).
    df["bateria_por_hora_tela"] = df["battery_drain_mah"] / df["screen_on_hours"]

    # Minutos de uso de app por hora de tela (intensidade de interação).
    df["uso_por_hora_tela"] = df["app_usage_min"] / df["screen_on_hours"]

    # Faixa etária (categórica) para análises demográficas.
    df["faixa_etaria"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 60],
        labels=["18-25", "26-35", "36-45", "46-59"],
    )

    # Rótulo textual da classe de comportamento.
    df["behavior_label"] = df["behavior_class"].map(config.BEHAVIOR_LABELS)

    return df


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de limpeza e engenharia de atributos.

    Args:
        df_raw: DataFrame bruto vindo de ``data_loader.load_raw_data``.

    Returns:
        DataFrame analítico pronto para estatística e modelagem.
    """
    df = _padronizar_colunas(df_raw)

    # Validações defensivas (o dataset é limpo, mas o código não assume isso).
    n_nulos = int(df.isnull().sum().sum())
    n_dups = int(df.duplicated(subset="user_id").sum())
    if n_dups:
        df = df.drop_duplicates(subset="user_id")
    print(f"[preprocessing] Nulos: {n_nulos} | duplicatas removidas: {n_dups}")

    df = _criar_features(df)
    df = df.reset_index(drop=True)
    print(f"[preprocessing] Base analítica final: {df.shape[0]:,} linhas × {df.shape[1]} colunas.")
    return df


def save_processed(df: pd.DataFrame) -> None:
    """Persiste a base limpa em Parquet (com fallback para CSV)."""
    config.ensure_directories()
    try:
        df.to_parquet(config.PROCESSED_FILE, index=False)
        print(f"[preprocessing] Base salva em {config.PROCESSED_FILE}")
    except Exception as exc:  # pragma: no cover
        fallback = config.PROCESSED_FILE.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        print(f"[preprocessing] Parquet indisponível ({exc}); base salva em {fallback}")


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    df_limpo = clean_data(load_raw_data())
    save_processed(df_limpo)
    print(df_limpo.head())
