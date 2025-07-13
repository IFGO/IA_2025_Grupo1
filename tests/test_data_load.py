import pytest
import pandas as pd
from modules.data_load import load_all_cryptos


class TestDataLoad:
    """Casos de teste para o módulo data_load."""

    def test_load_all_cryptos_expected_keys(self):
        """Testa que load_all_cryptos retorna as chaves de criptomoedas esperadas."""
        expected_cryptos = ["ADA", "AVAX", "BNB", "BTC", "DOGE", "DOT", "ETH", "SHIB", "SOL", "XRP"]
        result = load_all_cryptos()

        for crypto in expected_cryptos:
            assert crypto in result
            assert isinstance(result[crypto], pd.DataFrame)

    def test_dataframe_date_index(self):
        """Testa que os DataFrames carregados têm o índice de data esperado."""
        result = load_all_cryptos()

        for crypto, df in result.items():
            assert df.index.name == 'date'
            assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_handles_missing_files_gracefully(self):
        """Testa que a função trata arquivos ausentes graciosamente e continua o processamento."""
        result = load_all_cryptos(base_path="nonexistent_directory/")

        assert isinstance(result, dict)
        assert len(result) == 0
