import os
import pandas as pd

from typing import Dict
from modules.logging import get_logger


logger = get_logger('data_loading')


def load_all_cryptos(base_path: str = "data/") -> Dict[str, pd.DataFrame]:
    """
    Carrega os arquivos CSV das criptomoedas e organiza em um dicionário.

    Args:
        base_path (str): Caminho para a pasta com os arquivos .csv

    Returns:
        Dict[str, pd.DataFrame]: Dicionário contendo os DataFrames por criptomoeda
    """
    cryptos = ["ADA", "AVAX", "BNB", "BTC", "DOGE", "DOT", "ETH", "SHIB", "SOL", "XRP"]
    data = {}

    for coin in cryptos:
        file_path = os.path.join(base_path, f"{coin}.csv")
        try:
            df = pd.read_csv(file_path, skiprows=1)
            df = df.iloc[::-1].copy()
            df.columns = [col.lower() for col in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values(by='date', inplace=True)  # garante a ordem cronologica dos dados
            df.set_index('date', inplace=True)

            data[coin] = df
            logger.info(f"{coin} carregada com sucesso de {file_path}.")

        except Exception as e:
            logger.error(f"Erro ao carregar {coin} de {file_path}: {e}")

    return data
