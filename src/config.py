"""Configuração central do projeto: caminhos, parâmetros e constantes.

Centralizar a configuração evita "números mágicos" espalhados pelo código e
facilita a reprodutibilidade — um princípio básico de engenharia de software.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos do projeto (resolvidos de forma relativa à raiz do repositório)
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RAW_FILE: Path = RAW_DATA_DIR / "user_behavior_dataset.csv"
PROCESSED_FILE: Path = PROCESSED_DATA_DIR / "comportamento_limpo.parquet"

# --------------------------------------------------------------------------- #
# Parâmetros de análise
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# Detecção de outliers
IQR_MULTIPLIER: float = 1.5
ZSCORE_THRESHOLD: float = 3.0

# Estatística inferencial
ALPHA: float = 0.05

# Modelagem
KMEANS_N_CLUSTERS: int = 5     # esperamos recuperar as 5 classes de comportamento
TEST_SIZE: float = 0.25

# Colunas numéricas de interesse (nomes já padronizados pós-limpeza)
NUMERIC_COLS: list[str] = [
    "app_usage_min",
    "screen_on_hours",
    "battery_drain_mah",
    "num_apps",
    "data_usage_mb",
    "age",
]

# Mapeamento da classe de comportamento (alvo ordinal 1..5)
BEHAVIOR_LABELS: dict[int, str] = {
    1: "1 - Muito leve",
    2: "2 - Leve",
    3: "3 - Moderado",
    4: "4 - Pesado",
    5: "5 - Muito pesado",
}


def ensure_directories() -> None:
    """Garante que os diretórios de saída existam antes da escrita de artefatos."""
    for directory in (PROCESSED_DATA_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
