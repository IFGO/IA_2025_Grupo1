import pandas as pd
import os

from modules.visualizations import (
    salvar_boxplot_precos,
    salvar_histograma_precos,
    salvar_linha_media_mediana_moda
)


def test_boxplot_creation(tmp_path):
    data_dict = {
        'BTC': pd.DataFrame({'close': list(range(10))}),
        'ETH': pd.DataFrame({'close': list(range(10, 20))})
    }
    out = str(tmp_path)
    salvar_boxplot_precos(data_dict, out, dpi=250)
    assert os.path.exists(os.path.join(out, "boxplot_fechamento.png"))


def test_histogram_creation(tmp_path):
    data_dict = {
        'TEST': pd.DataFrame({'close': list(range(100))})
    }
    out = str(tmp_path)
    salvar_histograma_precos(data_dict, out, dpi=250)
    assert os.path.exists(os.path.join(out, "histograma_fechamento.png"))


def test_line_graph_creation(tmp_path):
    data_dict = {
        'TEST': pd.DataFrame({
            'close': list(range(30))
        }, index=pd.date_range("2023-01-01", periods=30))
    }
    out = str(tmp_path)
    salvar_linha_media_mediana_moda(data_dict, "TEST", out, dpi=250)
    assert os.path.exists(os.path.join(out, "TEST_linha_resumo.png"))
