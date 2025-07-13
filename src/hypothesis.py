from modules.data_load import load_all_cryptos
from modules.hypothesis_tests import perform_mean_return_monthly_test


def run_test(expected_return: float) -> None:
    """
    Executa o teste de hipótese sobre o retorno esperado médio.

    Args:
        expected_return (float): Valor esperado do retorno diário médio.
    """
    dados = load_all_cryptos(base_path="data/")

    print(f"Teste de hipótese com retorno esperado > {expected_return * 100:.2f}%:\n")
    resultados = perform_mean_return_monthly_test(dados, threshold_percent=expected_return)

    for moeda, (media, t, p, rejeita) in resultados.items():
        status = "✅ Rejeita H₀" if rejeita else "❌ Não rejeita H₀"
        print(f"{moeda:5}: média = {media:.4f}%, t = {t}, p = {p}, {status}")
