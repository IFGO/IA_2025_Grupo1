import numpy as np
import pandas as pd

from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from modules.logging import get_logger


logger = get_logger('models')


def prepare_features(df: pd.DataFrame, window: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria features baseadas em janelas móveis para previsão.

    Args:
        df (pd.DataFrame): DataFrame com os dados da criptomoeda.
        window (int): Número de dias anteriores para usar como feature.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) e target (y).
    """
    try:
        df = df.copy()
        for i in range(1, window + 1):
            df[f'lag_{i}'] = df['close'].shift(i)
        df.dropna(inplace=True)
        X = df[[f'lag_{i}' for i in range(1, window + 1)]].values
        y = df['close'].values
        return X, y
    except Exception as e:
        logger.exception("Erro ao preparar features: %s", e)
        raise


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em treino e teste.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Target.
        test_size (float): Proporção de teste (ex: 0.2 = 20%).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


def train_mlp_model(X: np.ndarray, y: np.ndarray, k: int = 5) -> Optional[MLPRegressor]:
    """
    Treina o modelo MLPRegressor usando validação cruzada K-Fold.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Target.
        k (int): Número de folds para K-Fold cross-validation.

    Returns:
        Optional[MLPRegressor]: Modelo treinado com melhor desempenho.
    """
    try:
        best_model = None
        best_score = float('inf')
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                                 solver='adam', max_iter=500, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            logger.info(f"Fold {fold+1}: MSE = {mse:.4f}, R² = {r2:.4f}")

            if mse < best_score:
                best_score = mse
                best_model = model

        logger.info("Treinamento concluído. Melhor MSE: %.4f", best_score)
        return best_model

    except Exception as e:
        logger.exception("Erro ao treinar modelo: %s", e)
        return None


def evaluate_model(model: MLPRegressor, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Avalia o modelo MLPRegressor usando MSE e R².

    Args:
        model (MLPRegressor): Modelo treinado.
        X_test (np.ndarray): Conjunto de teste (features).
        y_test (np.ndarray): Conjunto de teste (target).

    Returns:
        dict: Dicionário com métricas 'mse' e 'r2'.
    """
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Avaliação do modelo: MSE={mse:.4f}, R²={r2:.4f}")
        return {"mse": mse, "r2": r2}
    except Exception as e:
        logger.exception("Erro ao avaliar modelo: %s", e)
        raise


def train_linear_model(X: np.ndarray, y: np.ndarray) -> object:
    """
    Treina um modelo de regressão linear.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Target.

    Returns:
        object: Modelo treinado.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_polynomial_models(X: np.ndarray, y: np.ndarray, degrees: list = list(range(2, 11))) -> dict:
    """
    Treina modelos de regressão polinomial de grau 2 a 10.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Target.
        degrees (list): Lista de graus polinomiais.

    Returns:
        dict: Dicionário com os modelos treinados por grau.
    """
    models = {}
    for d in degrees:
        model = make_pipeline(PolynomialFeatures(d), LinearRegression())
        model.fit(X, y)
        models[d] = model
    return models
