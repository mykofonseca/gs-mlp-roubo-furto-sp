"""Inferência da MLP escolhida usando só NumPy.

A aplicação web carrega os arquivos de modelo/ e repete a propagação direta
(forward) da rede, sem precisar instalar o PyTorch no servidor.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessamento import PreProcessador

PASTA_PADRAO = Path(__file__).resolve().parents[1] / "modelo"


class ClassificadorOcorrencia:
    def __init__(self, pasta=PASTA_PADRAO):
        pasta = Path(pasta)
        with open(pasta / "metadados.json", encoding="utf-8") as arq:
            self.metadados = json.load(arq)
        self.prep = PreProcessador.carregar(pasta / "preprocessador.json")
        pesos = np.load(pasta / "pesos_mlp.npz")
        n = len([k for k in pesos.files if k.startswith("W")])
        self.camadas = [(pesos[f"W{i}"], pesos[f"b{i}"]) for i in range(n)]
        self.ativacao = self.metadados["arquitetura"]["ativacao"]

    def _ativar(self, z):
        return np.maximum(z, 0) if self.ativacao == "relu" else np.tanh(z)

    def probabilidade_roubo(self, df):
        a = self.prep.transformar(df)
        for i, (w, b) in enumerate(self.camadas):
            a = a @ w + b
            if i < len(self.camadas) - 1:
                a = self._ativar(a)
        return 1.0 / (1.0 + np.exp(-a[:, 0]))

    def classificar(self, registro):
        """Recebe um dicionário com os atributos e devolve classe e probabilidades."""
        p_roubo = float(self.probabilidade_roubo(pd.DataFrame([registro]))[0])
        limiar = self.metadados.get("limiar", 0.5)
        return {
            "classe": "roubo" if p_roubo >= limiar else "furto",
            "prob_roubo": p_roubo,
            "prob_furto": 1.0 - p_roubo,
        }
