import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt


def plot_histograms_features(df, features):
    nrows = len(features)
    fig, axs = plt.subplots(
        nrows=nrows, ncols=2, figsize=(12, 2.5 * nrows), layout="tight"
    )

    for idx, feat in enumerate(features):
        sns.histplot(data=df, x=feat, kde=True, ax=axs[idx, 0], color="navy")
        axs[idx, 0].set_title(f"{feat} Histogram")
        axs[idx, 0].axvline(x=df[feat].mean(), color="red", linestyle="--")
        sns.boxplot(data=df, x=feat, ax=axs[idx, 1], color="crimson")
        axs[idx, 1].set_title(f"{feat} Boxplot")

    plt.show()


def plot_corr_heatmap(df, figsize=(15, 7)):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="OrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.33,
        annot_kws={"fontsize": "x-small"},
        mask=mask,
    )
    plt.title("Correlation Matrix")


def plot_feature_importante(model, figsize=(10, 5)):
    # Pegando a importância das features
    importances = model.feature_importances_

    # Calculando a importância como porcentagem
    importances_percentage = (importances / importances.sum()) * 100

    # Plotando o gráfico
    fig, ax = plt.subplots(figsize=figsize)
    pd.Series(importances_percentage, index=model.feature_names_in_).sort_values(
        ascending=True
    ).plot(kind="barh", figsize=(15, 10))

    ax.set_title("Importância das Features (Porcentagem)")
    ax.set_xlabel("Importância Relativa (%)")

    # Ajustando os valores no eixo X para porcentagem
    x_ticks = ax.get_xticks()
    ax.set_xticklabels([f"{int(x)}%" for x in x_ticks])

    plt.show()


def calculate_ic(data, confidence=0.95):
    """
    Calcula o intervalo de confiança para uma amostra de dados.

    Parâmetros:
    data (array-like): Amostra de dados (lista, numpy array, etc.)
    confidence (float): Nível de confiança (ex.: 0.95 para 95%)

    Retorna:
    tuple: Intervalo de confiança (lower, upper)
    """
    # Converte para numpy array
    data = np.array(data)

    # Calcula a média e o desvio padrão
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # ddof=1 para usar a variância amostral
    n = len(data)

    # Calcula o valor crítico Z
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    # Calcula a margem de erro
    margin_of_error = z * (std_dev / np.sqrt(n))

    # Calcula o intervalo de confiança
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound


def plot_forecast(
    y_true,
    y_pred,
    title="Previsão de Temperatura",
    filter_start_date=None,
    xlabel="Data",
    ylabel="Temperatura (°C)",
    legend_labels=None,
):
    """
    Plota a previsão junto com a série temporal original.
    """

    # Filtrar data de início, se fornecida
    if filter_start_date:
        y_true = y_true[y_true.index.date > filter_start_date]
        y_pred = y_pred[y_pred.index.date > filter_start_date]

    # Plotar valores reais e previsões
    plt.figure(figsize=(10, 5))
    y_true.plot(label=legend_labels[0] if legend_labels else "Real", linewidth=1.5)
    y_pred.plot(label=legend_labels[1] if legend_labels else "Previsão", linewidth=1.5)

    # Adicionar legenda e grade
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(visible=True, linestyle="--", alpha=0.7)

    # Adicionar título e labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Ajustar ticks do eixo x
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Mostrar gráfico
    plt.show()
