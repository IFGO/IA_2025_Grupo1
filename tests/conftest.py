import sys
from pathlib import Path

# Adiciona o diretório src ao path do Python para habilitar a estrutura de diretórios
src_path = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(src_path))
