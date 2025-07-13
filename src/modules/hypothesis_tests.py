import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def perform_mean_return_test(
    data: Dict[str, pd.DataFrame],
    threshold_percent: float = 0.05,
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float, float, bool]]:
    """
    Realiza um teste de hipótese para verificar se o retorno médio diário
    é maior ou igual ao valor definido pelo usuário (em percentual).

    H0: μ >= x%
    H1: μ < x%

    Args:
        data (Dict[str, pd.DataFrame]): Dicionário de DataFrames com dados das criptomoedas.
        threshold_percent (float): Valor em percentual para comparação com a média dos retornos.
        alpha (float): Nível de significância (default: 0.05).

    Returns:
        Dict[str, Tuple[media_amostral, t_stat, p_value, rejeita_H0]]
    """

    results = {}

    for coin, df in data.items():
        try:
            df = df.sort_index()
            df['returns'] = df['close'].pct_change().dropna()

            returns = df['returns'].dropna() * 100  # aqui eu converto para percentual
            sample_mean = returns.mean()
            sample_std = returns.std(ddof=1)
            n = len(returns)

            # Testa a normalidade - Shapiro Wilk
            shapiro_stat, shapiro_p = stats.shapiro(returns)

            if shapiro_p >= alpha:
                # Para os dados que seguem uma distribuição normal — aplicar teste t
                t_stat = (sample_mean - threshold_percent) / (sample_std / np.sqrt(n))
                # p-value para teste bilateral a esquerda (menor que)
                p_value = stats.t.cdf(t_stat, df=n-1)
                reject_H0 = p_value < alpha
            else:
                # Para os dados que não seguem uma distribuição normal — não aplicar teste t
                print(f"[!] {coin}: distribuição dos retornos não é normal (p = {shapiro_p:.4f})")
                t_stat = np.nan
                p_value = np.nan
                reject_H0 = False

            results[coin] = (sample_mean, t_stat, p_value, reject_H0)

        except Exception as e:
            print(f"Erro na moeda {coin}: {e}")
            results[coin] = (np.nan, np.nan, np.nan, False)

    return results


def perform_mean_return_monthly_test(
    data: Dict[str, pd.DataFrame],
    threshold_percent: float = 0.2,
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float, float, bool]]:
    """
    Realiza um teste t de Student para verificar se o retorno médio mensal
    dos últimos 6 meses é maior ou igual a threshold_percent (%).

    Pré-condição: os dados devem passar no teste de normalidade de Shapiro-Wilk.

    H0: μ >= threshold_percent
    H1: μ < threshold_percent

    Args:
        data (Dict[str, pd.DataFrame]): Dicionário com DataFrames de criptomoedas.
        threshold_percent (float): Valor mínimo esperado de retorno mensal (%).
        alpha (float): Nível de significância.

    Returns:
        Dict[str, Tuple[media, t_stat, p_value, rejeita_H0]]
    """
    results = {}

    for coin, df in data.items():
        try:
            df = df.sort_index()
            df['date'] = df.index

            # Agrupa por mês e pega o último fechamento do mês
            df['month'] = df['date'].dt.to_period('M').apply(lambda r: r.start_time)
            monthly_close = df.groupby('month')['close'].last()

            # Calcula o retorno percentual mensal
            monthly_returns = monthly_close.pct_change().dropna() * 100

            # Considera apenas os últimos 6 meses
            monthly_returns = monthly_returns[-6:]

            # Verifica se há dados suficientes
            if len(monthly_returns) < 3:
                print(f"[!] {coin}: dados insuficientes para aplicar o teste (n={len(monthly_returns)})")
                results[coin] = (np.nan, np.nan, np.nan, False)
                continue

            _, shapiro_p = stats.shapiro(monthly_returns)

            if shapiro_p >= alpha:
                sample_mean = monthly_returns.mean()
                sample_std = monthly_returns.std(ddof=1)
                n = len(monthly_returns)

                t_stat = (sample_mean - threshold_percent) / (sample_std / np.sqrt(n))
                p_value = stats.t.cdf(t_stat, df=n - 1)
                reject_H0 = p_value < alpha
            else:
                print(f"[!] {coin}: distribuição mensal não é normal (p = {shapiro_p:.4f})")
                sample_mean = monthly_returns.mean()
                t_stat = np.nan
                p_value = np.nan
                reject_H0 = False

            results[coin] = (sample_mean, t_stat, p_value, reject_H0)

        except Exception as e:
            print(f"Erro na moeda {coin}: {e}")
            results[coin] = (np.nan, np.nan, np.nan, False)

    return results
