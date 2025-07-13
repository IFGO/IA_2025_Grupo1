import argparse
import sys
from pathlib import Path

# Adiciona o diretório src ao path do Python para habilitar a estrutura de diretórios
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import stats
import predict
import anova
import hypothesis


def run_stats(args: argparse.Namespace) -> None:
    """
    Executa a análise descritiva de todas as criptomoedas.

    Args:
        args (argparse.Namespace): Argumentos do parser contendo:
            - dpi (int): Resolução em DPI para os gráficos (padrão: 250)
    """
    stats.run_descriptive_analysis(dpi=args.dpi)


def run_predict(args: argparse.Namespace) -> None:
    """
    Executa a previsão do preço de fechamento usando modelo MLP.

    Args:
        args (argparse.Namespace): Argumentos do parser contendo:
            - crypto (str): Nome da criptomoeda
            - kfold (int): Número de folds para validação cruzada (padrão: 5)
            - window (int): Tamanho da janela para features temporais (padrão: 7)
            - compare (bool): Se deve executar comparação com outros modelos
    """
    predict.run_prediction(crypto=args.crypto, kfold=args.kfold, window=args.window)

    if args.compare:
        predict.run_comparison(crypto=args.crypto)


def run_anova(args: argparse.Namespace) -> None:
    """
    Executa a análise de variância (ANOVA) sobre os preços de fechamento.

    Args:
        args (argparse.Namespace): Argumentos do parser contendo:
            - period (str): Período de agregação ('W', 'ME', 'QE', padrão: 'ME')
            - window_size (int): Tamanho da janela para agregação (padrão: 6)
    """
    anova.run_analysis(period=args.period, window_size=args.window_size)


def run_hypothesis(args: argparse.Namespace) -> None:
    """
    Executa o teste de hipótese sobre o retorno esperado médio.

    Args:
        args (argparse.Namespace): Argumentos do parser contendo:
            - expected_return (float): Valor esperado do retorno diário médio (padrão: 0.2)
    """
    hypothesis.run_test(expected_return=args.expected_return)


def main():
    parser = argparse.ArgumentParser(description='Ferramenta para análise e previsão do preço de fechamento de criptomoedas')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Comando para análise descritiva
    stats_parser = subparsers.add_parser('stats', help='Gera estatísticas descritivas de todas as moedas')
    stats_parser.add_argument('--dpi', type=int, default=250, help='Resolução em DPI para os gráficos gerados (padrão: 250)')
    stats_parser.set_defaults(func=run_stats)

    # Comando para o modelo de previsão
    predict_parser = subparsers.add_parser('predict', help='Executa a previsão com o modelo MLP')
    predict_parser.add_argument('--crypto', type=str, required=True, help='Nome da criptomoeda (ex: BTC, ETH, etc.)')
    predict_parser.add_argument('--kfold', type=int, default=5, help='Número de folds para validação cruzada (padrão: 5)')
    predict_parser.add_argument('--window', type=int, default=7, help='Tamanho da janela para features de lag (padrão: 7)')
    predict_parser.add_argument(
        '--compare',
        action='store_true',
        default=False,
        help='Executa a comparação do modelo MLP com regressão (graus 1 a 10)'
    )
    predict_parser.set_defaults(func=run_predict)

    # Comando para análise de variância (ANOVA)
    anova_parser = subparsers.add_parser('anova', help='Executa a análise de variância sobre os preços de fechamento')
    anova_parser.add_argument(
        '--period',
        type=str,
        choices=['W', 'ME', 'QE'],
        default='ME',
        help='Período de agregação: W (Semanal), ME (Mensal), QE (Trimestral) (padrão: ME)'
    )
    anova_parser.add_argument(
        '--window-size',
        type=int,
        default=6,
        help='Tamanho da janela para agregação por período (padrão: 6)'
    )
    anova_parser.set_defaults(func=run_anova)

    # Comando para o teste de hipótese
    hypothesis_parser = subparsers.add_parser('hypothesis', help='Executa o teste de hipótese sobre o retorno esperado médio')
    hypothesis_parser.add_argument(
        '--expected-return',
        type=float,
        default=0.2,
        help='Valor esperado do retorno diário médio (padrão: 0.2)'
    )
    hypothesis_parser.set_defaults(func=run_hypothesis)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
