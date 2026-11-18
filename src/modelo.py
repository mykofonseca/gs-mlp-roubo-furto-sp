"""Definição, treino e avaliação das MLPs em PyTorch."""
import copy
import time

import numpy as np
import torch
from torch import nn

ATIVACOES = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Rede densa com saída de 1 neurônio (logit da classe roubo)."""

    def __init__(self, n_entradas, camadas, ativacao="relu", dropout=0.0):
        super().__init__()
        blocos, anterior = [], n_entradas
        for n in camadas:
            blocos += [nn.Linear(anterior, n), ATIVACOES[ativacao]()]
            if dropout > 0:
                blocos.append(nn.Dropout(dropout))
            anterior = n
        blocos.append(nn.Linear(anterior, 1))
        self.rede = nn.Sequential(*blocos)
        self.camadas = list(camadas)
        self.ativacao = ativacao

    def forward(self, x):
        # A sigmoide fica dentro da BCEWithLogitsLoss, que é numericamente mais estável
        return self.rede(x).squeeze(1)


def metricas(y_real, prob, limiar=0.5):
    y_pred = (prob >= limiar).astype(int)
    vp = int(((y_pred == 1) & (y_real == 1)).sum())
    vn = int(((y_pred == 0) & (y_real == 0)).sum())
    fp = int(((y_pred == 1) & (y_real == 0)).sum())
    fn = int(((y_pred == 0) & (y_real == 1)).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "acuracia": (vp + vn) / len(y_real),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matriz": [[vn, fp], [fn, vp]],
    }


def prever_proba(modelo, x):
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(torch.as_tensor(x))).numpy()


def treinar(
    x_tr, y_tr, x_val, y_val, camadas, ativacao="relu", dropout=0.0, lr=1e-3,
    weight_decay=0.0, epocas=60, batch_size=256, paciencia=None, semente=42,
    peso_positivo=None,
):
    """Treina uma MLP e devolve o modelo com os pesos da melhor época de validação.

    Com paciencia=None a rede roda todas as épocas (útil para observar overfitting);
    caso contrário, para quando a perda de validação não melhora por `paciencia` épocas.
    """
    torch.manual_seed(semente)
    np.random.seed(semente)
    modelo = MLP(x_tr.shape[1], camadas, ativacao, dropout)
    pos_weight = torch.tensor(peso_positivo) if peso_positivo else None
    perda_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)

    xt, yt = torch.as_tensor(x_tr), torch.as_tensor(y_tr, dtype=torch.float32)
    xv, yv = torch.as_tensor(x_val), torch.as_tensor(y_val, dtype=torch.float32)
    gerador = torch.Generator().manual_seed(semente)

    historico = {"perda_treino": [], "perda_val": [], "acc_treino": [], "acc_val": []}
    melhor = {"perda": float("inf"), "epoca": 0, "pesos": None}
    inicio = time.perf_counter()

    for epoca in range(1, epocas + 1):
        modelo.train()
        ordem = torch.randperm(len(xt), generator=gerador)
        for i in range(0, len(xt), batch_size):
            lote = ordem[i:i + batch_size]
            otimizador.zero_grad()
            perda = perda_fn(modelo(xt[lote]), yt[lote])
            perda.backward()
            otimizador.step()

        modelo.eval()
        with torch.no_grad():
            logit_tr, logit_val = modelo(xt), modelo(xv)
            historico["perda_treino"].append(perda_fn(logit_tr, yt).item())
            historico["perda_val"].append(perda_fn(logit_val, yv).item())
            historico["acc_treino"].append(((logit_tr > 0).float() == yt).float().mean().item())
            historico["acc_val"].append(((logit_val > 0).float() == yv).float().mean().item())

        if historico["perda_val"][-1] < melhor["perda"]:
            melhor = {"perda": historico["perda_val"][-1], "epoca": epoca,
                      "pesos": copy.deepcopy(modelo.state_dict())}
        elif paciencia and epoca - melhor["epoca"] >= paciencia:
            break

    modelo.load_state_dict(melhor["pesos"])
    info = {
        "melhor_epoca": melhor["epoca"],
        "epocas_executadas": len(historico["perda_val"]),
        "tempo_s": time.perf_counter() - inicio,
        "parametros": sum(p.numel() for p in modelo.parameters()),
    }
    return modelo, historico, info


def exportar_pesos(modelo, caminho):
    """Salva pesos e vieses das camadas lineares em .npz para a inferência em NumPy."""
    lineares = [m for m in modelo.rede if isinstance(m, nn.Linear)]
    arrays = {}
    for i, camada in enumerate(lineares):
        arrays[f"W{i}"] = camada.weight.detach().numpy().T
        arrays[f"b{i}"] = camada.bias.detach().numpy()
    np.savez(caminho, **arrays)
