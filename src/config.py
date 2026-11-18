"""Caminhos e constantes compartilhados pelo projeto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

URL_BASE_SSP = (
    "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/baseDados/"
    "celularesSub/CelularesSubtraidos_2025.xlsx"
)
PAGINA_SSP = "https://www.ssp.sp.gov.br/estatistica/consultas"

ARQ_BRUTO = RAIZ / "data" / "raw" / "CelularesSubtraidos_2025.xlsx"
ARQ_PARQUET = RAIZ / "data" / "raw" / "celulares_2025.parquet"
ARQ_PROCESSADO = RAIZ / "data" / "processed" / "ocorrencias_sp_2025.csv"
ARQ_RESUMO = RAIZ / "data" / "processed" / "resumo_preparacao.json"

PASTA_MODELO = RAIZ / "modelo"
PASTA_FIGURAS = RAIZ / "docs" / "img"

SEMENTE = 42
TAMANHO_AMOSTRA = 20_000

# Atributos de entrada da MLP (nenhum deles indica a natureza jurídica)
CATEGORICAS = ["delegacia", "periodo", "dia_semana", "tipo_local", "subtipo_local", "marca"]
NUMERICAS = ["qtd_celulares"]
ALVO = "roubo"
