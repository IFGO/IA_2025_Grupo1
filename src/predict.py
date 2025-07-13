import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from modules.logging import get_logger
from modules.data_load import load_all_cryptos
from modules.simulation import simulate_profit, simulate_profit_series
from modules.visualizations import plot_real_vs_pred, plot_balance_evolution
from modules.models import (
    prepare_features,
    train_mlp_model,
    evaluate_model,
    split_data,
    train_linear_model,
    train_polynomial_models,
)


logger = get_logger('prediction')


def run_prediction(crypto: str, kfold: int, window: int) -> None:
    """
    Executa o pipeline de carregamento, processamento, treinamento e avaliação
    de um modelo MLP para previsão do preço de fechamento de uma criptomoeda.
    """
    try:
        # Carrega os dados
        dados = load_all_cryptos()

        if crypto not in dados:
            logger.error(f"Criptomoeda {crypto} não encontrada no dataset.")
            print(f"Erro: Criptomoeda {crypto} não disponível.")
            return

        df = dados[crypto]
        if 'close' not in df.columns:
            logger.error(f"Coluna 'close' não encontrada nos dados de {crypto}.")
            print("Erro: Coluna 'close' ausente no arquivo.")
            return

        # Exibe resumo da Crypto
        resumo = df['close'].agg(['mean', 'std', 'min', 'max'])
        print(f"\nResumo estatístico para {crypto}:")
        print(f"Média: ${resumo['mean']:.2f}")
        print(f"Desvio Padrão: ${resumo['std']:.2f}")
        print(f"Mínimo: ${resumo['min']:.2f}")
        print(f"Máximo: ${resumo['max']:.2f}")
        print()

        # Prepara os dados
        X, y = prepare_features(df, window=window)

        # Divide em treino/teste (80/20)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        # Treina o modelo com K-Fold no treino
        model = train_mlp_model(X_train, y_train, k=kfold)

        if model is None:
            print("Erro durante o treinamento do modelo.")
            return

        # Avalia o modelo
        metrics = evaluate_model(model, X_test, y_test)
        y_pred = model.predict(X_test)

        # Plota gráfico Real vs. Previsto
        plot_real_vs_pred(y_test, y_pred, title=f"{crypto} - Real vs. Previsto")
        print(f"\nAvaliação do modelo MLP para {crypto}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"R² : {metrics['r2']:.4f}")

        # Simulação de lucro
        final_balance = simulate_profit(y_test, y_pred, initial_balance=1000.0)
        print("Valores reais:", y_test[:5])
        print("Previsões:", y_pred[:5])
        print(f"\n Simulação de lucro: saldo final = ${final_balance:,.2f}")

    except Exception as e:
        logger.exception("Erro na execução principal: %s", e)
        print(f"Erro durante a execução: {e}")


def run_comparison(crypto: str):
    dados = load_all_cryptos()
    df = dados[crypto]

    # Prepara os dados
    X, y = prepare_features(df, window=7)
    X_train, X_test = X[: -30], X[-30:]
    y_train, y_test = y[: -30], y[-30:]

    # Treinando os modelos
    mlp = train_mlp_model(X_train, y_train, k=5)
    linear = train_linear_model(X_train, y_train)
    poly_models = train_polynomial_models(X_train, y_train)

    # Fazendo as previsões
    preds = {
        "MLP": mlp.predict(X_test),
        "Linear": linear.predict(X_test),
    }
    for grau, model in poly_models.items():
        preds[f"Poly_{grau}"] = model.predict(X_test)

    # Simulando o lucro diário
    saldos = {
        nome: simulate_profit_series(y_test, y_pred, 1000.0)
        for nome, y_pred in preds.items()
    }

    # Adicionando a curva real da estratégia Buy & Hold
    # saldos["Buy & Hold"] = simulate_hold_strategy(y_test)

    # Gráfico da evolução do saldo
    plot_balance_evolution(saldos, title="Evolução do Lucro - Modelos")

    # Diagrama de dispersão
    plt.figure(figsize=(10, 6))
    for nome, y_pred in preds.items():
        plt.scatter(y_test, y_pred, label=nome, alpha=0.6)
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title("Diagrama de Dispersão - Todos os Modelos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/diagrama_dispersao_modelos.png", dpi=150)
    plt.show()

    # Métricas e equações
    print("\nComparação dos Modelos:\n")
    comparacoes = []
    for nome, y_pred in preds.items():
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        std_error = np.std(y_test - y_pred)
        corr, _ = pearsonr(y_test, y_pred)

        if "Poly" not in nome:
            coef, intercept = np.polyfit(y_test, y_pred, 1)
            eq = f"y = {coef:.4f} * x + {intercept:.4f}"
        else:
            eq = "Equação polinomial"

        print(f"Modelo: {nome}")
        print(f"  - MSE          : {mse:.4f}")
        print(f"  - R²           : {r2:.4f}")
        print(f"  - Correlação   : {corr:.4f}")
        print(f"  - Equação      : {eq}")
        print(f"  - Erro Padrão  : {std_error:.4f}\n")

        comparacoes.append((nome, mse, std_error))

    # Fazendo a comparação final entre MLP e o melhor modelo
    comparacoes.sort(key=lambda x: x[1])  # ordena por MSE
    mlp_std = [c[2] for c in comparacoes if c[0] == "MLP"][0]
    melhor = [c for c in comparacoes if c[0] != "MLP"][0]
    print(f"Diferença entre MLP e {melhor[0]} (erro padrão): {mlp_std - melhor[2]:.4f}")
