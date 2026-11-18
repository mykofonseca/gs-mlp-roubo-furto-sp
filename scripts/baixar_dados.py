"""Baixa a base da SSP-SP e gera a amostra processada usada no notebook.

Uso (na raiz do repositório):  python scripts/baixar_dados.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.dados import baixar_base, preparar_base  # noqa: E402

if __name__ == "__main__":
    print(f"Baixando {config.URL_BASE_SSP} ...")
    caminho = baixar_base()
    print(f"Arquivo salvo em {caminho}. Lendo a planilha (leva alguns minutos)...")
    _, resumo = preparar_base()
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(f"Amostra salva em {config.ARQ_PROCESSADO}")
