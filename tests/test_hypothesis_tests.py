import pandas as pd
import numpy as np
from modules.hypothesis_tests import (
    perform_mean_return_test,
    perform_mean_return_monthly_test
)


class TestHypothesisTests:
    """Testes para as funções de testes de hipóteses estatísticas."""

    def setup_method(self):
        """Configura dados de teste para cada método."""
        np.random.seed(42)

        # Criar dados simulados de criptomoedas
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # BTC com retornos positivos
        btc_prices = 50000 + np.cumsum(np.random.normal(0.1, 1000, 100))
        self.btc_df = pd.DataFrame({'close': btc_prices}, index=dates)

        # ETH com retornos negativos
        eth_prices = 3000 + np.cumsum(np.random.normal(-0.05, 100, 100))
        self.eth_df = pd.DataFrame({'close': eth_prices}, index=dates)

        # Dicionário de dados
        self.data_dict = {
            'BTC': self.btc_df,
            'ETH': self.eth_df
        }

    def test_test_mean_return_basic(self):
        """Testa o teste básico de retorno médio."""
        result = perform_mean_return_test(self.data_dict, threshold_percent=0.05)

        assert isinstance(result, dict)
        assert 'BTC' in result
        assert 'ETH' in result

        # Verificar estrutura da tupla de resultado
        for coin, (mean, t_stat, p_value, reject) in result.items():
            assert isinstance(mean, (float, np.floating))

    def test_test_mean_return_different_thresholds(self):
        """Testa diferentes thresholds para o teste."""
        for threshold in [0.01, 0.05, 0.1]:
            result = perform_mean_return_test(self.data_dict, threshold_percent=threshold)
            assert isinstance(result, dict)
            assert len(result) == 2  # BTC e ETH

    def test_test_mean_return_monthly_basic(self):
        """Testa o teste mensal básico."""
        result = perform_mean_return_monthly_test(self.data_dict, threshold_percent=0.2)

        assert isinstance(result, dict)
        assert 'BTC' in result
        assert 'ETH' in result

        # Verificar estrutura da tupla de resultado
        for coin, (mean, t_stat, p_value, reject) in result.items():
            assert isinstance(mean, (float, np.floating))

    def test_test_mean_return_monthly_different_thresholds(self):
        """Testa diferentes thresholds para o teste mensal."""
        for threshold in [0.1, 0.2, 0.5]:
            result = perform_mean_return_monthly_test(self.data_dict, threshold_percent=threshold)
            assert isinstance(result, dict)
            assert len(result) == 2

    def test_test_mean_return_empty_data(self):
        """Testa comportamento com dados vazios."""
        empty_dict = {}
        result = perform_mean_return_test(empty_dict)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_test_mean_return_monthly_empty_data(self):
        """Testa comportamento mensal com dados vazios."""
        empty_dict = {}
        result = perform_mean_return_monthly_test(empty_dict)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_test_mean_return_insufficient_data(self):
        """Testa comportamento com dados insuficientes."""
        # Criar dados com apenas 2 pontos (insuficiente para teste)
        short_dates = pd.date_range('2023-01-01', periods=2, freq='D')
        short_prices = [1000, 1010]
        short_df = pd.DataFrame({'close': short_prices}, index=short_dates)

        short_dict = {'SHORT': short_df}
        result = perform_mean_return_test(short_dict)

        assert isinstance(result, dict)
        assert 'SHORT' in result
