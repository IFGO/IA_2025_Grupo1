import numpy as np
from modules.simulation import simulate_profit


class TestSimulation:
    """Testes para as funções de simulação de lucro."""

    def setup_method(self):
        """Configura dados de teste para cada método."""
        np.random.seed(42)

        # Criar dados simulados de preços reais e previstos
        self.y_true = np.array([100, 101, 99, 102, 103, 98, 104, 105, 97, 106])
        self.y_pred = np.array([100.5, 100.8, 99.2, 101.5, 102.8, 98.5, 103.2, 104.5, 97.8, 105.2])

        # Dados com tendência de alta
        self.y_true_up = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
        self.y_pred_up = np.array([101, 103, 105, 107, 109, 111, 113, 115, 117, 119])

        # Dados com tendência de baixa
        self.y_true_down = np.array([100, 98, 96, 94, 92, 90, 88, 86, 84, 82])
        self.y_pred_down = np.array([99, 97, 95, 93, 91, 89, 87, 85, 83, 81])

    def test_simulate_profit_basic(self):
        """Testa a simulação básica de lucro."""
        initial_balance = 1000.0
        final_balance = simulate_profit(self.y_true, self.y_pred, initial_balance)

        assert isinstance(final_balance, float)
        assert final_balance >= 0  # Saldo final não pode ser negativo
        assert final_balance != initial_balance  # Deve haver alguma mudança

    def test_simulate_profit_different_initial_balances(self):
        """Testa diferentes saldos iniciais."""
        for initial_balance in [100, 1000, 10000]:
            final_balance = simulate_profit(self.y_true, self.y_pred, initial_balance)
            assert isinstance(final_balance, float)
            assert final_balance >= 0

    def test_simulate_profit_upward_trend(self):
        """Testa simulação com tendência de alta."""
        initial_balance = 1000.0
        final_balance = simulate_profit(self.y_true_up, self.y_pred_up, initial_balance)

        # Em tendência de alta, deve haver lucro
        assert final_balance > initial_balance

    def test_simulate_profit_downward_trend(self):
        """Testa simulação com tendência de baixa."""
        initial_balance = 1000.0
        final_balance = simulate_profit(self.y_true_down, self.y_pred_down, initial_balance)

        # Em tendência de baixa, pode haver perda
        assert isinstance(final_balance, float)
        assert final_balance >= 0

    def test_simulate_profit_perfect_predictions(self):
        """Testa simulação com predições perfeitas."""
        # Predições idênticas aos valores reais
        y_true = np.array([100, 101, 102, 103, 104])
        y_pred = np.array([100, 101, 102, 103, 104])

        initial_balance = 1000.0
        final_balance = simulate_profit(y_true, y_pred, initial_balance)

        # Com predições perfeitas, deve haver lucro máximo
        assert final_balance > initial_balance

    def test_simulate_profit_opposite_predictions(self):
        """Testa simulação com predições opostas."""
        # Predições opostas aos valores reais
        y_true = np.array([100, 101, 102, 103, 104])
        y_pred = np.array([104, 103, 102, 101, 100])

        initial_balance = 1000.0
        final_balance = simulate_profit(y_true, y_pred, initial_balance)

        assert final_balance > initial_balance

    def test_simulate_profit_constant_prices(self):
        """Testa simulação com preços constantes."""
        # Preços constantes
        y_true = np.array([100, 100, 100, 100, 100])
        y_pred = np.array([100, 100, 100, 100, 100])

        initial_balance = 1000.0
        final_balance = simulate_profit(y_true, y_pred, initial_balance)

        # Com preços constantes, não deve haver mudança
        assert final_balance == initial_balance

    def test_simulate_profit_single_data_point(self):
        """Testa simulação com apenas um ponto de dados."""
        y_true = np.array([100])
        y_pred = np.array([101])

        initial_balance = 1000.0
        final_balance = simulate_profit(y_true, y_pred, initial_balance)

        assert isinstance(final_balance, float)
        assert final_balance >= 0

    def test_simulate_profit_negative_prices(self):
        """Testa comportamento com preços negativos."""
        y_true = np.array([-100, -101, -102])
        y_pred = np.array([-99, -100, -101])

        initial_balance = 1000.0
        final_balance = simulate_profit(y_true, y_pred, initial_balance)

        assert isinstance(final_balance, float)
        assert final_balance >= 0

    def test_simulate_profit_high_volatility(self):
        """Testa simulação com alta volatilidade."""
        # Dados com alta volatilidade
        y_true = np.array([100, 120, 80, 140, 60, 160, 40, 180, 20, 200])
        y_pred = np.array([110, 130, 90, 150, 70, 170, 50, 190, 30, 210])

        initial_balance = 1000.0
        final_balance = simulate_profit(y_true, y_pred, initial_balance)

        assert isinstance(final_balance, float)
        assert final_balance >= 0

    def test_simulate_profit_consistency(self):
        """Testa consistência da simulação."""
        # Executar múltiplas vezes com os mesmos dados
        initial_balance = 1000.0
        results = []

        for _ in range(5):
            final_balance = simulate_profit(self.y_true, self.y_pred, initial_balance)
            results.append(final_balance)

        # Todos os resultados devem ser iguais (determinístico)
        assert len(set(results)) == 1
