"""Testes rápidos do pré-processamento e da inferência usada pela aplicação."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.dados import periodo_do_dia  # noqa: E402
from src.inferencia import ClassificadorOcorrencia  # noqa: E402
from src.preprocessamento import PreProcessador, dividir  # noqa: E402


def test_periodo_do_dia():
    assert periodo_do_dia("05:59:00", None) == "madrugada"
    assert periodo_do_dia("06:00:00", None) == "manha"
    assert periodo_do_dia("12:30:00", None) == "tarde"
    assert periodo_do_dia("23:10:00", None) == "noite"
    assert periodo_do_dia(None, "A noite") == "noite"
    assert periodo_do_dia(None, "Em hora incerta") is None


def test_divisao_sem_sobreposicao():
    df = pd.read_csv(config.ARQ_PROCESSADO)
    partes = dividir(df, config.ALVO, config.SEMENTE)
    assert sum(len(p) for p in partes.values()) == len(df)
    taxas = [p[config.ALVO].mean() for p in partes.values()]
    assert max(taxas) - min(taxas) < 0.01


def test_categoria_desconhecida_vira_zeros():
    treino = pd.DataFrame({"cor": ["a", "b"], "n": [1.0, 3.0]})
    prep = PreProcessador(["cor"], ["n"]).ajustar(treino)
    x = prep.transformar(pd.DataFrame({"cor": ["z"], "n": [2.0]}))
    assert x.tolist() == [[0.0, 0.0, 0.0]]


def test_classificador_devolve_probabilidades_validas():
    clf = ClassificadorOcorrencia()
    registro = {
        "delegacia": "47º D.P. CAPAO REDONDO", "periodo": "noite", "dia_semana": "sexta",
        "tipo_local": "Via Pública", "subtipo_local": "Via Pública", "marca": "Apple", "qtd_celulares": 1,
    }
    r = clf.classificar(registro)
    assert r["classe"] in {"roubo", "furto"}
    assert np.isclose(r["prob_roubo"] + r["prob_furto"], 1.0)


def test_numpy_igual_pytorch():
    import torch

    from src.modelo import MLP

    clf = ClassificadorOcorrencia()
    arq = clf.metadados["arquitetura"]
    modelo = MLP(arq["entradas"], arq["camadas_ocultas"], arq["ativacao"], arq["dropout"])
    modelo.load_state_dict(torch.load(config.PASTA_MODELO / "mlp_roubo_furto.pt"))
    modelo.eval()
    df = pd.read_csv(config.ARQ_PROCESSADO).head(500)
    with torch.no_grad():
        p_torch = torch.sigmoid(modelo(torch.as_tensor(clf.prep.transformar(df)))).numpy()
    assert np.abs(p_torch - clf.probabilidade_roubo(df)).max() < 1e-5
