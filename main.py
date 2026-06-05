"""Pipeline completo — orquestra todas as etapas da análise.

Uso:
    python main.py
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data_loader import load_raw_data
from src.descriptive_stats import resumo_categorico, tabela_descritiva, teste_normalidade
from src.eda_advanced import (
    analise_bivariada_alvo,
    matriz_correlacao,
    perfil_por_classe,
    perfil_por_so,
    resumo_outliers,
)
from src.inferential_stats import (
    anova_classes,
    qui_quadrado,
    tabela_intervalos_confianca,
    teste_t_android_vs_ios,
    teste_t_genero,
)
from src.modeling import segmentar_kmeans, treinar_classificador, treinar_regressao_linear
from src.preprocessing import clean_data, save_processed
from src import visualization as viz


def _cabecalho(titulo: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {titulo}")
    print("=" * 72)


def executar_pipeline() -> pd.DataFrame:
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    config.ensure_directories()

    _cabecalho("ETAPA 1-2 | CARREGAMENTO E LIMPEZA")
    df = clean_data(load_raw_data())
    save_processed(df)

    _cabecalho("ETAPA 3 | ESTATÍSTICA DESCRITIVA")
    print(tabela_descritiva(df))
    print("\n-- Normalidade --")
    print(teste_normalidade(df))
    print("\n-- Sistema operacional --")
    print(resumo_categorico(df, "os"))

    _cabecalho("ETAPA 4 | EDA AVANÇADA")
    print(resumo_outliers(df))
    print("\n-- Correlação com a classe de comportamento --")
    print(analise_bivariada_alvo(df))
    print("\n-- Perfil médio por classe --")
    print(perfil_por_classe(df))
    print("\n-- Android vs iOS --")
    print(perfil_por_so(df))

    _cabecalho("ETAPA 5 | ESTATÍSTICA INFERENCIAL")
    print(teste_t_android_vs_ios(df))
    print(teste_t_genero(df))
    print(anova_classes(df))
    print(qui_quadrado(df))
    print("\n-- Intervalos de confiança (95%) --")
    print(tabela_intervalos_confianca(df))

    _cabecalho("ETAPA 6 | MODELAGEM")
    reg = treinar_regressao_linear(df)
    print(f"Regressão OLS (bateria) | R² treino={reg.r2_treino:.3f} | teste={reg.r2_teste:.3f}")
    print(reg.coeficientes.round(4))
    clf = treinar_classificador(df)
    print(f"\nClassificação (perfil) | acurácia={clf.acuracia:.3f}")
    print(clf.relatorio)
    clu = segmentar_kmeans(df)
    print(f"K-Means | silhouette={clu.silhouette:.3f} | Rand ajustado={clu.rand_ajustado:.3f}")
    print(clu.perfil_clusters)

    _cabecalho("GERAÇÃO DE FIGURAS")
    viz.plot_distribuicoes(df, salvar_como="01_distribuicoes.png")
    viz.plot_boxplots_outliers(df, salvar_como="02_boxplots_outliers.png")
    viz.plot_matriz_correlacao(matriz_correlacao(df, "pearson"),
                               "Matriz de correlação (Pearson)", salvar_como="03_correlacao.png")
    viz.plot_uso_por_classe(df, "app_usage_min", salvar_como="04_uso_por_classe.png")
    viz.plot_android_vs_ios(df, "battery_drain_mah", salvar_como="05_android_vs_ios.png")
    viz.plot_matriz_confusao(clf.matriz_confusao, salvar_como="06_matriz_confusao.png")
    viz.plot_clusters(clu.df_clusterizado, salvar_como="07_clusters_kmeans.png")

    _cabecalho("PIPELINE CONCLUÍDO")
    print(f"Figuras em: {config.FIGURES_DIR}")
    return df


if __name__ == "__main__":
    executar_pipeline()
