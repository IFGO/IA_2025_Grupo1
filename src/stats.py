import pandas as pd

from modules.data_load import load_all_cryptos
from modules.logging import get_logger
from modules.visualizations import (
    salvar_boxplot_precos,
    salvar_histograma_precos,
    salvar_linha_media_mediana_moda,
    salvar_multiplos_graficos_linha,
    salvar_grafico_variabilidade
)

logger = get_logger('stats')


def analise_estatistica(df: pd.DataFrame) -> None:
    """
    Exibe e registra medidas estatísticas das criptomoedas.

    Args:
        df (pd.DataFrame): DataFrame contendo dados de todas as criptomoedas.
    """
    logger.info("Iniciando análise estatística...")

    # Cálculo das estatísticas
    resumo = df.groupby('Cripto')['close'].agg(
        media='mean',
        mediana='median',
        std='std',
        variancia='var',
        minimo='min',
        maximo='max'
    ).reset_index()

    # Cálculo de amplitude e coeficiente de variação (CV)
    resumo['amplitude'] = resumo['maximo'] - resumo['minimo']
    resumo['coef_var'] = resumo['std'] / resumo['media']

    # Salvar em CSV
    print("Salvando resumo da variabilidade em um arquivo csv...")
    resumo.to_csv("figures/resumo_estatistico.csv", index=False)

    # Gráfico de variabilidade
    salvar_grafico_variabilidade(df, "figures")

    # Impressão com formatação
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print("\nResumo estatístico por Criptomoeda:")
        print(resumo[['Cripto', 'media', 'mediana', 'std', 'variancia', 'amplitude', 'coef_var']])

        print("\nTop 3 moedas com maior desvio padrão:")
        print(resumo.sort_values(by='std', ascending=False)[['Cripto', 'std']].head(3))

        print("\nTop 3 moedas com maior amplitude:")
        print(resumo.sort_values(by='amplitude', ascending=False)[['Cripto', 'amplitude']].head(3))

        print("\nTop 3 moedas com maior coeficiente de variação (CV):")
        print(resumo.sort_values(by='coef_var', ascending=False)[['Cripto', 'coef_var']].head(3))

    logger.info("Resumo estatístico calculado e exibido com sucesso.")


def run_descriptive_analysis(dpi: int) -> None:
    """
    Executa a análise descritiva de todas as criptomoedas.
    Imrpime as estatísticas e salva os gráficos no diretório figures/.

    Args:
        dpi (int): Resolução em DPI para os gráficos.
    """
    try:
        dados = load_all_cryptos("data/")
        logger.info(f"Chaves disponíveis: {dados.keys()}")

        df_all = pd.concat([
            df.assign(Cripto=coin) for coin, df in dados.items()
        ])

        analise_estatistica(df_all)
        logger.info("Análise estatística concluída!")

        print('\nGerando gráficos da análise...')

        salvar_boxplot_precos(dados, 'figures', dpi=dpi)
        salvar_histograma_precos(dados, 'figures', dpi=dpi)

        for crypto in dados.keys():
            logger.info(f"Gerando gráfico para {crypto}...")
            salvar_linha_media_mediana_moda(dados, crypto, 'figures/cryptos', dpi=dpi)

        salvar_multiplos_graficos_linha(dados, 'figures', dpi=dpi)
        print('Gráficos gerados em figures/')

    except Exception:
        print("Erro durante a análise descritiva")
        logger.exception("Erro durante a análise descritiva")
