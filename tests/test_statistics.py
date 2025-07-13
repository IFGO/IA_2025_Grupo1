import pandas as pd
import numpy as np
from anova import calculate_avg_daily_returns


class TestStatistics:
    """Testes para as funções estatísticas."""

    def setup_method(self):
        """Configura dados de teste para cada método."""
        # Criar dados simulados para diferentes criptomoedas
        np.random.seed(42)

        # Datas comuns
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Simular preços de fechamento com tendência e volatilidade
        btc_prices = 50000 + np.cumsum(np.random.normal(0, 1000, 100))
        eth_prices = 3000 + np.cumsum(np.random.normal(0, 100, 100))
        ada_prices = 1 + np.cumsum(np.random.normal(0, 0.05, 100))

        # Criar DataFrames
        self.btc_df = pd.DataFrame({
            'close': btc_prices
        }, index=dates)

        self.eth_df = pd.DataFrame({
            'close': eth_prices
        }, index=dates)

        self.ada_df = pd.DataFrame({
            'close': ada_prices
        }, index=dates)

        # Dicionário de dados
        self.data_dict = {
            'BTC': self.btc_df,
            'ETH': self.eth_df,
            'ADA': self.ada_df
        }

    def test_calculate_avg_daily_returns_basic(self):
        """Testa o cálculo básico de retornos diários."""
        result = calculate_avg_daily_returns(self.data_dict)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3  # BTC, ETH, ADA
        assert all(col in result.columns for col in ['BTC', 'ETH', 'ADA'])
        assert len(result) > 0

        # Verificar se os retornos estão entre -1 e 1 (normalmente)
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

    def test_calculate_avg_daily_returns_structure(self):
        """Testa a estrutura do DataFrame de retornos."""
        result = calculate_avg_daily_returns(self.data_dict)

        # Verificar se não há valores NaN no início (após dropna)
        assert not result.iloc[0].isna().any()

        # Verificar se o índice é datetime
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_calculate_avg_daily_returns_alignment(self):
        """Testa se os dados são alinhados corretamente."""
        result = calculate_avg_daily_returns(self.data_dict)

        # Verificar se todas as colunas têm o mesmo número de linhas
        lengths = [len(result[col].dropna()) for col in result.columns]
        assert len(set(lengths)) == 1  # Todas devem ter o mesmo tamanho

    def test_calculate_avg_daily_returns_empty_dict(self):
        """Testa comportamento com dicionário vazio."""
        empty_dict = {}
        result = calculate_avg_daily_returns(empty_dict)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0
        assert len(result) == 0

    def test_calculate_avg_daily_returns_single_crypto(self):
        """Testa com apenas uma criptomoeda."""
        single_dict = {'BTC': self.btc_df}
        result = calculate_avg_daily_returns(single_dict)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1
        assert 'BTC' in result.columns
        assert len(result) > 0

    def test_calculate_avg_daily_returns_different_lengths(self):
        """Testa com criptomoedas de diferentes tamanhos."""
        # Criar dados com tamanhos diferentes
        short_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        long_dates = pd.date_range('2023-01-01', periods=100, freq='D')

        short_prices = 1000 + np.cumsum(np.random.normal(0, 10, 50))
        long_prices = 2000 + np.cumsum(np.random.normal(0, 20, 100))

        short_df = pd.DataFrame({'close': short_prices}, index=short_dates)
        long_df = pd.DataFrame({'close': long_prices}, index=long_dates)

        mixed_dict = {
            'SHORT': short_df,
            'LONG': long_df
        }

        result = calculate_avg_daily_returns(mixed_dict)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 49  # Deve usar o tamanho da menor série

    def test_calculate_avg_daily_returns_constant_prices(self):
        """Testa com preços constantes (retornos devem ser 0)."""
        constant_prices = np.full(100, 1000)
        constant_df = pd.DataFrame({
            'close': constant_prices
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))

        constant_dict = {'CONST': constant_df}
        result = calculate_avg_daily_returns(constant_dict)

        # Retornos devem ser 0 (exceto o primeiro que será NaN)
        assert result['CONST'].iloc[1:].abs().max() < 1e-10
