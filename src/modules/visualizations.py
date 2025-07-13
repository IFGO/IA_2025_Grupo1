import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import math

from typing import Dict
from modules.logging import get_logger


logger = get_logger('visualizations')


def plot_real_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Real vs. Previsto", dpi: int = 150) -> None:
    """
    Plota o gráfico do valor real vs. previsto pelo modelo.

    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.
        title (str): Título do gráfico.
        dpi (int): Resolução da imagem (dots per inch).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Real", linewidth=2)
    plt.plot(y_pred, label="Previsto", linestyle="--")
    plt.title(title)
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Preço de Fechamento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{title.lower().replace(' ', '_')}.png", dpi=dpi)
    plt.show()


def plot_balance_evolution(balance_dict: dict, title: str = "Evolução do Lucro", dpi: int = 150) -> None:
    """
    Plota a evolução do saldo diário para múltiplos modelos.

    Args:
        balance_dict (dict): Dicionário com nome do modelo como chave e lista de saldos como valor.
        title (str): Título do gráfico.
        dpi (int): Resolução da imagem (dots per inch).
    """
    plt.figure(figsize=(12, 6))

    for nome_modelo, saldo_diario in balance_dict.items():
        plt.plot(saldo_diario, label=nome_modelo)

    plt.title(title)
    plt.xlabel("Dias")
    plt.ylabel("Saldo ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/evolucao_saldo_modelos.png", dpi=dpi)
    plt.show()


def salvar_boxplot_precos(data_dict: dict, output_path: str, dpi: int) -> None:
    """
    Gera e salva um boxplot dos preços de fechamento por criptomoeda.

    Args:
        data_dict (dict): Dicionário com DataFrames das criptomoedas.
        output_path (str): Caminho para salvar a imagem.
        dpi (int): Resolução da imagem (dots per inch).
    """
    logger.info("Iniciando geração do boxplot...")
    logger.info(f"Criptomoedas disponíveis: {list(data_dict.keys())}")

    # Prepara dados para o boxplot
    boxplot_data = []
    crypto_names = []

    for crypto, df in data_dict.items():
        boxplot_data.extend(df['close'].values)
        crypto_names.extend([crypto] * len(df))

    logger.info(f"Total de dados coletados para o boxplot: {len(boxplot_data)}")

    df_boxplot = pd.DataFrame({
        'Crypto': crypto_names,
        'Close': boxplot_data
    })

    logger.info(f"DataFrame do boxplot criado com shape: {df_boxplot.shape}")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_boxplot, x='Crypto', y='Close')
    plt.title("Boxplot do Preço de Fechamento - 10 Criptomoedas")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    caminho_fig = os.path.join(output_path, "boxplot_fechamento.png")
    plt.savefig(caminho_fig, dpi=dpi)
    plt.close()

    logger.info(f"Boxplot salvo em: {caminho_fig}")


def salvar_histograma_precos(data_dict: dict, output_path: str, dpi: int) -> None:
    """
    Gera e salva um histograma dos preços de fechamento (todas as moedas).

    Args:
        data_dict (dict): Dicionário com DataFrames das criptomoedas.
        output_path (str): Caminho para salvar a imagem.
        dpi (int): Resolução da imagem (dots per inch).
    """
    # Prepara dados para o histograma
    all_prices = []
    for crypto, df in data_dict.items():
        all_prices.extend(df['close'].values)

    plt.figure(figsize=(10, 5))
    sns.histplot(data=all_prices, kde=True, bins=50)
    plt.title("Distribuição dos Preços de Fechamento")
    plt.xlabel("Preço")
    plt.ylabel("Frequência")
    plt.grid(True)

    os.makedirs(output_path, exist_ok=True)
    caminho_fig = os.path.join(output_path, "histograma_fechamento.png")
    plt.savefig(caminho_fig, dpi=dpi)
    plt.close()

    logger.info(f"Histograma salvo em: {caminho_fig}")


def salvar_linha_media_mediana_moda(data_dict: dict, cripto: str, output_path: str, dpi: int) -> None:
    """
    Gera e salva gráfico de linha com preço de fechamento, média, mediana e moda.

    Args:
        data_dict (dict): Dicionário com DataFrames das criptomoedas.
        cripto (str): Nome da criptomoeda.
        output_path (str): Pasta para salvar o gráfico.
        dpi (int): Resolução da imagem (dots per inch).
    """
    df = data_dict[cripto].copy()
    df['media_7d'] = df['close'].rolling(7).mean()
    df['mediana_7d'] = df['close'].rolling(7).median()
    moda = df['close'].mode().iloc[0]
    df['moda'] = moda

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label="Fechamento")
    plt.plot(df.index, df['media_7d'], label="Média 7d")
    plt.plot(df.index, df['mediana_7d'], label="Mediana 7d")
    plt.axhline(moda, color='gray', linestyle='--', label=f"Moda: {moda:.2f}")
    plt.title(f"{cripto} - Fechamento com Média, Mediana e Moda")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_path, exist_ok=True)
    caminho_fig = os.path.join(output_path, f"{cripto}_linha_resumo.png")
    plt.savefig(caminho_fig, dpi=dpi)
    plt.close()

    logger.info(f"Gráfico linha salvo para {cripto} em: {caminho_fig}")


def salvar_multiplos_graficos_linha(dados: Dict[str, pd.DataFrame], output_path: str, dpi: int) -> None:
    """
    Gera um único gráfico com subplots (1 por criptomoeda) mostrando:
    - preço de fechamento
    - média móvel 7d
    - mediana móvel 7d
    - moda

    Args:
        dados (Dict[str, pd.DataFrame]): Dicionário com os dados de cada cripto
        output_path (str): Pasta para salvar o gráfico
        dpi (int): Resolução da imagem (dots per inch).
    """

    n = len(dados)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=False)
    axes = axes.flatten()

    for i, (coin, df) in enumerate(dados.items()):
        ax = axes[i]
        df_plot = df.copy()
        df_plot['media_7d'] = df_plot['close'].rolling(7).mean()
        df_plot['mediana_7d'] = df_plot['close'].rolling(7).median()
        moda = df_plot['close'].mode().iloc[0]

        ax.plot(df_plot.index, df_plot['close'], label='Fechamento')
        ax.plot(df_plot.index, df_plot['media_7d'], label='Média 7d')
        ax.plot(df_plot.index, df_plot['mediana_7d'], label='Mediana 7d')
        ax.axhline(moda, color='gray', linestyle='--', label=f'Moda: {moda:.2f}')
        ax.set_title(coin)
        ax.legend()
        ax.grid(True)

    # Remove subplots vazios (caso len(dados) não seja múltiplo de cols)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, "linhas_resumo_todas.png")
    plt.savefig(fig_path, dpi=dpi)
    plt.close()

    logger.info(f"Subplots salvos em {fig_path}")


def salvar_grafico_variabilidade(df_all: pd.DataFrame, output_path: str) -> None:
    """
    Salva gráfico de barras com o desvio padrão de cada criptomoeda usando escala logarítmica.

    Args:
        df_all (pd.DataFrame): DataFrame com coluna 'Cripto' e 'close'
        output_path (str): Pasta onde salvar o gráfico
    """

    try:
        variabilidade = df_all.groupby("Cripto")["close"].std()
        variabilidade = variabilidade[variabilidade > 0].sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=variabilidade.index, y=variabilidade.values, ax=ax)
        ax.set_yscale("log")  # ✅ ESCALA LOG APLICADA CORRETAMENTE
        ax.set_title("Variabilidade (Desvio Padrão) por Criptomoeda - Escala Log")
        ax.set_ylabel("Desvio Padrão (log)")
        ax.set_xlabel("Criptomoeda")
        ax.grid(True)

        os.makedirs(output_path, exist_ok=True)
        fig_path = os.path.join(output_path, "variabilidade_barras.png")
        fig.savefig(fig_path, dpi=150)
        plt.close()

        logger.info(f"Gráfico de variabilidade com escala logarítmica salvo em {fig_path}")
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de variabilidade: {e}")
