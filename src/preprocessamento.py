"""Codificação one-hot e padronização dos atributos.

Implementado sem scikit-learn para que os mesmos parâmetros, salvos em JSON,
sejam reaproveitados pela aplicação web sem depender do ambiente de treino.
"""
import json

import numpy as np


class PreProcessador:
    def __init__(self, categoricas, numericas):
        self.categoricas = list(categoricas)
        self.numericas = list(numericas)
        self.categorias = {}
        self.media = {}
        self.desvio = {}

    def ajustar(self, df):
        """Aprende categorias, média e desvio. Deve receber apenas o conjunto de treino."""
        for col in self.categoricas:
            self.categorias[col] = sorted(df[col].astype(str).unique().tolist())
        for col in self.numericas:
            valores = df[col].astype(float)
            self.media[col] = float(valores.mean())
            self.desvio[col] = float(valores.std(ddof=0)) or 1.0
        return self

    @property
    def nomes_saida(self):
        nomes = [f"{c}={v}" for c in self.categoricas for v in self.categorias[c]]
        return nomes + list(self.numericas)

    def transformar(self, df):
        blocos = []
        for col in self.categoricas:
            valores = df[col].astype(str).to_numpy()
            cats = np.array(self.categorias[col])
            # Categoria não vista no treino vira um vetor só de zeros
            blocos.append((valores[:, None] == cats[None, :]).astype(np.float32))
        for col in self.numericas:
            z = (df[col].astype(float).to_numpy() - self.media[col]) / self.desvio[col]
            blocos.append(z.astype(np.float32)[:, None])
        return np.hstack(blocos)

    def para_dict(self):
        return {
            "categoricas": self.categoricas,
            "numericas": self.numericas,
            "categorias": self.categorias,
            "media": self.media,
            "desvio": self.desvio,
        }

    @classmethod
    def de_dict(cls, d):
        obj = cls(d["categoricas"], d["numericas"])
        obj.categorias, obj.media, obj.desvio = d["categorias"], d["media"], d["desvio"]
        return obj

    def salvar(self, caminho):
        with open(caminho, "w", encoding="utf-8") as arq:
            json.dump(self.para_dict(), arq, ensure_ascii=False, indent=2)

    @classmethod
    def carregar(cls, caminho):
        with open(caminho, encoding="utf-8") as arq:
            return cls.de_dict(json.load(arq))


def dividir(df, alvo, semente, frac_val=0.15, frac_teste=0.15):
    """Divisão estratificada em treino, validação e teste (70/15/15)."""
    rng = np.random.default_rng(semente)
    partes = {"treino": [], "validacao": [], "teste": []}
    for _, grupo in df.groupby(alvo):
        idx = rng.permutation(grupo.index.to_numpy())
        n_teste = round(len(idx) * frac_teste)
        n_val = round(len(idx) * frac_val)
        partes["teste"].append(idx[:n_teste])
        partes["validacao"].append(idx[n_teste:n_teste + n_val])
        partes["treino"].append(idx[n_teste + n_val:])
    return {k: df.loc[np.sort(np.concatenate(v))].reset_index(drop=True) for k, v in partes.items()}

