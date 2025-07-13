import numpy as np

from modules.logging import get_logger


logger = get_logger('simulation')


def simulate_profit(y_true: np.ndarray, y_pred: np.ndarray, initial_balance: float = 1000.0) -> float:
    """
    Simula o lucro com reinvestimento diário baseado na previsão do modelo.

    A lógica é:
        - Compra hoje se previsão do próximo dia for maior que o valor de hoje.
        - Vende no dia seguinte, e reinveste o saldo total.

    Args:
        y_true (np.ndarray): Valores reais de fechamento.
        y_pred (np.ndarray): Valores previstos de fechamento.
        initial_balance (float): Saldo inicial em USD.

    Returns:
        float: Saldo final ao fim da simulação.
    """

    balance = initial_balance
    min_price = 1.0  # Valor mínimo razoável para considerar como preço real (evita divisões explosivas)

    for i in range(len(y_true) - 1):
        today_real = y_true[i]
        tomorrow_real = y_true[i + 1]
        tomorrow_pred = y_pred[i + 1]

        # Verificações de sanidade
        if (np.isnan(today_real) or np.isnan(tomorrow_real) or np.isnan(tomorrow_pred)):
            continue
        if (today_real <= min_price or tomorrow_real <= min_price):
            continue
        if tomorrow_pred > today_real:
            change = tomorrow_real / today_real

            # Se a mudança for absurda (> 10x em um dia), ignorar
            if change > 10:
                logger.warning(f"Variação anormal ignorada: {change:.2f}x no dia {i}")
                continue

            balance *= change

    return round(balance, 2)


def simulate_profit_series(y_true: np.ndarray, y_pred: np.ndarray, initial_balance: float = 1000.0) -> list:
    """
    Retorna uma lista com o saldo acumulado dia a dia baseado nas previsões.
    Compra/venda ocorre apenas se a previsão do próximo dia for maior que o valor atual.

    Args:
        y_true (np.ndarray): Valores reais do ativo.
        y_pred (np.ndarray): Valores previstos pelo modelo.
        initial_balance (float): Valor inicial em dólares.

    Returns:
        list: Lista com o saldo em cada dia.
    """
    balance = initial_balance
    balances = [balance]
    min_price = 1.0

    for i in range(len(y_true) - 1):
        today_real = y_true[i]
        tomorrow_real = y_true[i + 1]
        tomorrow_pred = y_pred[i + 1]

        if np.isnan(today_real) or np.isnan(tomorrow_real) or np.isnan(tomorrow_pred):
            balances.append(balance)
            continue

        if today_real <= min_price or tomorrow_real <= min_price:
            balances.append(balance)
            continue

        if tomorrow_pred > today_real:
            change = tomorrow_real / today_real
            if change > 10:
                balances.append(balance)
                continue
            balance *= change

        balances.append(balance)

    return balances


def simulate_hold_strategy(y_true: np.ndarray, initial_balance: float = 1000.0) -> np.ndarray:
    """
    Simula a estratégia Buy and Hold com base nos preços reais.

    Args:
        y_true (np.ndarray): Preços reais de fechamento.
        initial_balance (float): Saldo inicial.

    Returns:
        np.ndarray: Vetor com saldo acumulado ao longo dos dias.
    """
    if len(y_true) < 2:
        return np.array([initial_balance])

    preco_inicial = y_true[0]
    quantidade = initial_balance / preco_inicial
    saldo_diario = quantidade * y_true
    return saldo_diario
