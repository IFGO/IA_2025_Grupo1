# Projeto de Análise de Dados de Criptomoedas

Este projeto implementa um sistema de previsão do preço de fechamento de criptomoedas utilizando modelos de
aprendizado de máquina, com foco principal no MLPRegressor. Ele também compara esse modelo com regrssão linear
e polinomial (grau 2 a 10), incluindo simulações de lucro com reinvestimento diário.

## 1. Estrutura do Projeto

```
projeto_final/
├── main.py                     # Script principal com interface de linha de comando
├── requirements.txt            # Dependências do projeto
├── .flake8                    # Configuração do flake8
├── pytest.ini                 # Configuração do pytest
├── data/                      # Arquivos CSV das criptomoedas
├── figures/                   # Gráficos gerados pelos comandos
├── logs/                      # Arquivos de log
├── tests/                     # Testes automatizados
│   ├── test_data_load.py
│   ├── test_models.py
│   ├── test_simulation.py
│   ├── test_visualizations.py
│   ├── test_hypothesis_tests.py
│   ├── test_statistics.py
│   └── conftest.py
├── src/                       # Código fonte principal
│   ├── predict.py             # Comando de previsão MLP
│   ├── stats.py               # Comando de análise descritiva
│   ├── anova.py               # Comando de análise ANOVA
│   ├── hypothesis.py          # Comando de teste de hipótese
│   └── modules/               # Módulos compartilhados
│       ├── data_load.py       # Carregamento de dados
│       ├── models.py          # Modelos de ML (MLP, Linear, Polinomial)
│       ├── simulation.py      # Simulação de lucro
│       ├── visualizations.py  # Geração de gráficos
│       ├── hypothesis_tests.py # Testes de hipótese
│       └── logging.py         # Configuração de logs
└── htmlcov/                   # Relatórios de cobertura de testes
```

## 2. Instalação

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

Recomendamos que o projeto seja executado com o python 3.11.13.

## 3. Como Usar

O projeto oferece quatro comandos principais através do `main.py`:

### Análise Descritiva (`stats`)

Gera estatísticas descritivas e gráficos para todas as criptomoedas:

```bash
python main.py stats
```

**Parâmetros:**
- `--dpi` (opcional): Resolução em DPI para os gráficos (padrão: 250)

**Gráficos gerados em `figures/`:**
- `boxplot_fechamento.png`: Boxplot dos preços de fechamento por criptomoeda
- `histograma_fechamento.png`: Distribuição dos preços de fechamento
- `cryptos/[CRYPTO]_linha_resumo.png`: Gráfico de linha com média, mediana e moda para cada cripto
- `linhas_resumo_todas.png`: Subplots com todos os gráficos de linha

**Exemplo:**
```bash
python main.py stats --dpi 300
```

### Previsão com Modelo MLP (`predict`)

Executa previsão do preço de fechamento usando MLPRegressor:

```bash
python main.py predict --crypto BTC
```

**Parâmetros obrigatórios:**
- `--crypto`: Nome da criptomoeda (ex: BTC, ETH, ADA, AVAX, BNB, DOGE, DOT, SHIB, SOL, XRP)

**Parâmetros opcionais:**
- `--kfold`: Número de folds para validação cruzada (padrão: 5)
- `--window`: Tamanho da janela para features temporais (padrão: 7)
- `--compare`: Executa comparação com regressão linear e polinomial (graus 2 a 10)

**Gráficos gerados em `figures/`:**
- `[CRYPTO]_real_vs_previsto.png`: Gráfico real vs previsto
- `evolucao_saldo_modelos.png`: Evolução do saldo dos modelos (com `--compare`)
- `diagrama_dispersao_modelos.png`: Diagrama de dispersão dos modelos (com `--compare`)

**Exemplos:**
```bash
# Previsão básica
python main.py predict --crypto ETH

# Com parâmetros customizados
python main.py predict --crypto BTC --kfold 10 --window 5

# Com comparação de modelos
python main.py predict --crypto ETH --compare
```

### Análise de Variância (`anova`)

Executa análise ANOVA sobre os preços de fechamento:

```bash
python main.py anova --period ME
```

**Parâmetros obrigatórios:**
- `--period`: Período de agregação
  - `W`: Semanal
  - `ME`: Mensal (padrão)
  - `QE`: Trimestral

**Parâmetros opcionais:**
- `--window-size`: Tamanho da janela para agregação (padrão: 6)

**Saída:**
- Resultados dos testes de normalidade
- Resultados dos testes de variância
- Análise ANOVA completa
- Análise por volume de trades (alto/baixo)

**Exemplos:**
```bash
# Análise mensal (padrão)
python main.py anova

# Análise semanal
python main.py anova --period W --window-size 8

# Análise trimestral
python main.py anova --period QE --window-size 4
```

### Teste de Hipótese (`hypothesis`)

Executa teste de hipótese sobre o retorno esperado médio:

```bash
python main.py hypothesis
```

**Parâmetros opcionais:**
- `--expected-return`: Valor esperado do retorno diário médio (padrão: 0.2)

**Saída:**
- Resultados do teste de hipótese para cada criptomoeda
- Estatísticas t, p-valores e decisão de rejeição

**Exemplos:**
```bash
# Teste com valor padrão
python main.py hypothesis

# Teste com valor customizado
python main.py hypothesis --expected-return 0.15
```

## 7. Executando Testes

Execute todos os testes com cobertura:

```bash
pytest
```

Execute um arquivo de teste específico:

```bash
pytest tests/test_data_load.py
```

Execute um caso de teste específico:

```bash
pytest tests/test_data_load.py::test_load_all_cryptos_expected_keys
```

### Verificação de Estilo do Código com Flake8

O projeto utiliza o **flake8** para verificar a consistência do estilo do código Python.

Há um arquivo `.flake8` que define algumas regras específicas ao projeto:
- Comprimento máximo de linha: 130 caracteres
- Exclusão de diretórios: `.git`, `__pycache__`

De modo geral, usamos a configuração padrão do `flake`. Algumas restrições que valem ser destacadas:

- Indentação de 4 espaços (não tabs)
- Espaçamento adequado ao redor de operadores e após vírgulas
- Linhas em branco adequadas entre funções e classes (2 linhas)
- Evitar muitas linhas em branco consecutivas
- Imports devem estar no topo do arquivo
- Não usar imports não utilizados
- Evitar `from module import *`

#### Executando o Flake8

```bash
flake8
```

## 8. Relatórios de Cobertura

Após executar os testes, os relatórios de cobertura são gerados automaticamente:

### Relatório de Cobertura HTML
Abra `htmlcov/index.html` no seu navegador web para visualizar o relatório detalhado de cobertura.
