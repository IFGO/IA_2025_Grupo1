import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from typing import Dict, Tuple
from statsmodels.formula.api import ols
from modules.data_load import load_all_cryptos


def calculate_avg_daily_returns(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calcula os retornos diários médios para todas as criptomoedas.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Dicionário com DataFrames das criptomoedas.

    Returns:
        pd.DataFrame: DataFrame com os retornos diários de todas as criptomoedas.
    """
    returns_df = pd.DataFrame()

    for crypto, df in data_dict.items():
        returns_df[crypto] = df['close'].pct_change()

    return returns_df.dropna()


def calculate_avg_trade_count(data_dict: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Calcula a média do volume de trades por criptomoeda.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Dicionário com DataFrames das criptomoedas.

    Returns:
        pd.Series: Série com a média de trades por criptomoeda.
    """
    trade_counts = pd.DataFrame()

    for crypto, df in data_dict.items():
        trade_counts[crypto] = df.resample('ME')['tradecount'].mean()

    return trade_counts.tail(12).mean()


def check_normalities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica a normalidade dos dados usando teste de Shapiro-Wilk.

    Args:
        df (pd.DataFrame): DataFrame com os dados a serem testados.

    Returns:
        pd.DataFrame: DataFrame com resultados do teste de normalidade por criptomoeda.
    """
    results = []

    for crypto in df.columns:
        _, p_value = stats.shapiro(df[crypto])
        results.append({'crypto': crypto, 'p_value': p_value, 'is_normal': p_value > 0.05})

    return pd.DataFrame(results).set_index('crypto')


def check_homoscedasticity(df: pd.Series | pd.DataFrame) -> Tuple[bool, float]:
    """
    Verifica a homoscedasticidade (igualdade de variâncias) usando teste de Levene.

    Args:
        df (pd.Series | pd.DataFrame): Dados para verificar homoscedasticidade.

    Returns:
        Tuple[bool, float]: (é_homoscedástico, p_value).
    """
    all_cryptos = [df[crypto] for crypto in df.columns]
    _, p_lev = stats.levene(*all_cryptos, center='median')

    return (p_lev > 0.05, p_lev)


def evaluate_anova_premises(df: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Avalia as premissas necessárias para ANOVA: normalidade e homoscedasticidade.

    Args:
        df (pd.DataFrame): DataFrame com os dados para análise.

    Returns:
        pd.Series | pd.DataFrame: DataFrame filtrado apenas com criptomoedas que atendem às premissas.
    """
    normalities = check_normalities(df)

    print('Normalidade de cada criptomoeda:')
    print(normalities)

    normal_cryptos = normalities[normalities['is_normal']].index.tolist()
    filtered_df = df[normal_cryptos]

    print(f'\nCriptomoedas com distribuição normal: {normal_cryptos}')
    print(f'Criptomoedas removidas: {[c for c in df.columns if c not in normal_cryptos]}')

    is_homoscedastic, p_lev = check_homoscedasticity(filtered_df)
    print(f'\nHomoscedasticidade: p-value: {p_lev:.4f}, Mesma variância? {is_homoscedastic}')

    return filtered_df


def run_anova_analysis(df: pd.Series | pd.DataFrame) -> None:
    """
    Executa a análise ANOVA e verifica a normalidade dos resíduos.

    Args:
        df (pd.Series | pd.DataFrame): DataFrame com os dados para análise ANOVA.
    """
    long_df = df.melt(var_name='crypto', value_name='quarterly_avg_return')

    print('\nMédia geral por criptomoeda:')
    print(long_df.groupby('crypto').mean())

    model = ols('quarterly_avg_return ~ crypto', data=long_df).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    print('\nANOVA:')
    print(anova)

    _, p_val_resid = stats.shapiro(model.resid)
    print(f'\nNormalidade dos resíduos: p-value: {p_val_resid:.4f}, Normal? {p_val_resid > 0.05}')


def run_analysis(period: str, window_size: int) -> None:
    """
    Executa a análise completa de ANOVA para criptomoedas.

    Args:
        period (str): Período de agregação ('W', 'ME', 'QE').
        window_size (int): Tamanho da janela para análise.
    """
    all_cryptos = load_all_cryptos()
    avg_df = calculate_avg_daily_returns(all_cryptos)
    agg_df = avg_df.resample(period).mean().tail(window_size)

    print(f'Agrupando os dados por {period} (últimos {window_size} {period}s)\n')
    df_for_anova = evaluate_anova_premises(agg_df)
    run_anova_analysis(df_for_anova)

    print('\nAgrupando a análise por volume de trades')
    avg_trade_count = calculate_avg_trade_count(all_cryptos)
    trade_count_median = avg_trade_count.median()
    high_trade_cryptos = avg_trade_count[avg_trade_count >= trade_count_median].index.tolist()
    low_trade_cryptos = avg_trade_count[avg_trade_count < trade_count_median].index.tolist()

    print(f'\nANOVA para volume de trades baixo: {low_trade_cryptos}')
    low_trade_df = evaluate_anova_premises(agg_df[low_trade_cryptos])
    run_anova_analysis(low_trade_df)

    print(f'\nANOVA para volume de trades alto: {high_trade_cryptos}')
    high_trade_df = evaluate_anova_premises(agg_df[high_trade_cryptos])
    run_anova_analysis(high_trade_df)
