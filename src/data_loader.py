"""Etapa 1 — Carregamento dos dados brutos.

Responsabilidade única: ler o CSV do disco. Toda a limpeza/transformação
acontece em ``preprocessing.py``.
"""
from __future__ import annotations

import pandas as pd

from src import config


def load_raw_data() -> pd.DataFrame:
    """Carrega o dataset bruto de comportamento de uso de smartphones.

    Returns:
        DataFrame com as colunas originais (nomes ainda não padronizados).
    """
    df = pd.read_csv(config.RAW_FILE)
    print(f"[data_loader] {len(df):,} registros carregados de {config.RAW_FILE.name}.")
    return df


if __name__ == "__main__":
    raw = load_raw_data()
    print(raw.head())
    print(raw.shape)
