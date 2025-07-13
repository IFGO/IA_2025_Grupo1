import pytest
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor

from modules.models import (
    prepare_features,
    train_mlp_model,
    evaluate_model,
    split_data
)


class TestModels:
    """Testes para as funções de modelos de machine learning."""

    def setup_method(self):
        """Configura dados de teste controlados para cada método."""
        # Criar dados controlados de preços de criptomoeda
        dates = pd.date_range('2023-01-01', periods=20, freq='D')

        # Preços com padrão previsível: crescente com pequenas variações
        prices = [100 + i * 2 + (i % 3) for i in range(20)]  # 100, 102, 105, 106, 109, ...

        self.df = pd.DataFrame({
            'close': prices,
            'open': [p - 1 for p in prices],
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'volume': [1000 + i * 50 for i in range(20)]
        }, index=dates)

    def test_prepare_features_basic(self):
        """Testa a preparação básica de features."""
        X, y = prepare_features(self.df, window=3)

        # Verificar se não há valores NaN
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

        # Verificar valores específicos (previsíveis)
        # Primeira linha: [106, 103, 100] -> target: 106
        assert X[0, 0] == 106  # lag_1
        assert X[0, 1] == 103  # lag_2
        assert X[0, 2] == 100  # lag_3
        assert y[0] == 106     # target

    def test_prepare_features_different_windows(self):
        """Testa diferentes tamanhos de janela."""
        test_cases = [
            (1, 19),  # window=1, expected_samples=19
            (3, 17),  # window=3, expected_samples=17
            (5, 15),  # window=5, expected_samples=15
            (10, 10)  # window=10, expected_samples=10
        ]

        for window, expected_samples in test_cases:
            X, y = prepare_features(self.df, window=window)

            assert X.shape[1] == window
            assert len(X) == len(y)
            assert len(X) == expected_samples

    def test_split_data_basic(self):
        """Testa o split básico dos dados."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4)

        assert len(X_train) == 3
        assert len(X_test) == 2
        assert len(y_train) == 3
        assert len(y_test) == 2

    def test_split_data_different_test_sizes(self):
        """Testa diferentes tamanhos de teste."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])

        test_cases = [
            (0.2, 4, 1),  # 20% test = 1 sample
            (0.4, 3, 2),  # 40% test = 2 samples
            (0.6, 2, 3)   # 60% test = 3 samples
        ]

        for test_size, expected_train, expected_test in test_cases:
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

            assert len(X_train) == expected_train
            assert len(X_test) == expected_test
            assert len(y_train) == expected_train
            assert len(y_test) == expected_test

    def test_train_mlp_model_basic(self):
        """Testa o treinamento básico do modelo MLP."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([3, 7, 11, 15, 19])  # y = x1 + x2

        model = train_mlp_model(X, y, k=2)

        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')

    def test_train_mlp_model_different_k(self):
        """Testa diferentes valores de k para validação cruzada."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([3, 7, 11, 15, 19])

        for k in [2, 3, 5]:
            model = train_mlp_model(X, y, k=k)
            assert model is not None

    def test_evaluate_model_basic(self):
        """Testa a avaliação básica do modelo."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([3, 7, 11, 15, 19])

        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)

        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0
        assert metrics['r2'] <= 1

    def test_evaluate_model_perfect_prediction(self):
        """Testa avaliação com predição próxima da perfeita."""
        # Dados onde y = x1 + x2 (relação linear)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([3, 7, 11, 15, 19])  # y = x1 + x2

        model = MLPRegressor(hidden_layer_sizes=(20,), max_iter=500, random_state=42)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)

        assert metrics['r2'] > 0.9
        assert metrics['mse'] < 2.0
