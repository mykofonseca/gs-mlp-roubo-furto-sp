"""Obtenção e preparação da base de celulares subtraídos da SSP-SP.

Fluxo: planilha bruta -> recorte da capital -> rótulo roubo/furto ->
uma linha por boletim -> atributos -> amostra de 20 mil boletins.
"""
import json
import urllib.request

import numpy as np
import pandas as pd

from src import config

RUBRICAS = {"Roubo (art. 157)": 1, "Furto (art. 155)": 0}
CHAVE_BO = ["ID_DELEGACIA", "ANO_BO", "NUM_BO"]
DIAS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
PERIODOS_SSP = {
    "De madrugada": "madrugada",
    "Pela manhã": "manha",
    "A tarde": "tarde",
    "A noite": "noite",
}
MIN_FREQ_LOCAL = 100
MIN_FREQ_MARCA = 500
MAX_CELULARES = 5


def baixar_base(destino=config.ARQ_BRUTO):
    """Baixa a planilha de 2025 do portal da SSP (cerca de 108 MB)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        return destino
    req = urllib.request.Request(config.URL_BASE_SSP, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(destino, "wb") as arq:
        while bloco := resp.read(1 << 20):
            arq.write(bloco)
    return destino


def carregar_bruto():
    """Lê a aba CELULAR_2025. A leitura do xlsx demora alguns minutos, então guardamos um parquet."""
    if config.ARQ_PARQUET.exists():
        return pd.read_parquet(config.ARQ_PARQUET)
    baixar_base()
    df = pd.read_excel(config.ARQ_BRUTO, sheet_name="CELULAR_2025", dtype=str)
    df.to_parquet(config.ARQ_PARQUET, index=False)
    return df


def periodo_do_dia(hora, periodo_ssp):
    """Unifica HORA_OCORRENCIA e DESCR_PERIODO em uma única categoria.

    Na base, cada boletim tem só um dos dois campos preenchidos, e qual deles
    aparece depende do sistema de registro. Usar a hora exata (ou o fato de ela
    estar ausente) deixaria a rede aprender o canal de registro, não o crime.
    """
    if isinstance(hora, str) and hora:
        h = int(hora[:2])
        if h < 6:
            return "madrugada"
        if h < 12:
            return "manha"
        if h < 18:
            return "tarde"
        return "noite"
    return PERIODOS_SSP.get(periodo_ssp)


def agrupar_raros(serie, minimo, rotulo="Outros"):
    contagem = serie.value_counts()
    raros = contagem[contagem < minimo].index
    return serie.where(~serie.isin(raros), rotulo)


def filtrar_capital(df):
    """Mantém fatos ocorridos na capital e só as rubricas de roubo e furto."""
    # NOME_MUNICIPIO é onde o BO foi registrado; o local do fato é a circunscrição
    capital = (df["NOME_MUNICIPIO_CIRC"] == "S.PAULO") & (df["CIDADE"] == "S.PAULO")
    return df[capital & df["RUBRICA"].isin(RUBRICAS)].copy()


def consolidar_boletins(df):
    """Reduz a base a uma linha por boletim.

    A base bruta repete o BO para cada celular, versão e natureza. Ficamos com a
    última versão, descartamos BOs com roubo e furto ao mesmo tempo e contamos
    quantos celulares distintos foram levados.
    """
    df["VERSAO"] = df["VERSAO"].astype(int)
    ultima = df.groupby(CHAVE_BO)["VERSAO"].transform("max")
    df = df[df["VERSAO"] == ultima]

    rubricas_por_bo = df.groupby(CHAVE_BO)["RUBRICA"].transform("nunique")
    ambiguos = df.loc[rubricas_por_bo > 1, CHAVE_BO].drop_duplicates()
    df = df[rubricas_por_bo == 1]

    qtd = df.groupby(CHAVE_BO)["CONT_OBJETO"].nunique().rename("qtd_celulares")
    df = df.drop_duplicates(CHAVE_BO).merge(qtd, on=CHAVE_BO)
    return df, len(ambiguos)


def montar_atributos(df):
    """Transforma colunas da SSP nos atributos usados pela MLP."""
    data = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")
    saida = pd.DataFrame(
        {
            "delegacia": df["NOME_DELEGACIA_CIRC"].str.strip(),
            "periodo": [periodo_do_dia(h, p) for h, p in zip(df["HORA_OCORRENCIA"], df["DESCR_PERIODO"])],
            "dia_semana": data.dt.dayofweek.map(dict(enumerate(DIAS))),
            "tipo_local": df["DESCR_TIPOLOCAL"],
            "subtipo_local": df["DESCR_SUBTIPOLOCAL"],
            "marca": df["MARCA_OBJETO"],
            "qtd_celulares": df["qtd_celulares"].clip(upper=MAX_CELULARES),
            "roubo": df["RUBRICA"].map(RUBRICAS),
            "ano_fato": data.dt.year,
        }
    )
    return saida


def limpar(df):
    """Remove registros incompletos e fora do recorte, agrupando categorias raras."""
    df = df[df["ano_fato"] == 2025].drop(columns="ano_fato")
    # Só delegacias territoriais (xx° D.P.); as especializadas não indicam região
    df = df[df["delegacia"].str.contains("D.P.", regex=False, na=False)]
    df = df.dropna(subset=["periodo", "dia_semana", "tipo_local", "subtipo_local", "marca"])
    df = df[(df["tipo_local"] != "NULL") & (df["subtipo_local"] != "NULL") & (df["marca"] != "NULL")]

    df["tipo_local"] = agrupar_raros(df["tipo_local"], MIN_FREQ_LOCAL)
    # Subtipo raro vira "Outros (<tipo>)" para continuar coerente com o tipo escolhido
    contagem = df["subtipo_local"].value_counts()
    raro = df["subtipo_local"].map(contagem) < MIN_FREQ_LOCAL
    df.loc[raro, "subtipo_local"] = "Outros (" + df.loc[raro, "tipo_local"] + ")"
    df["marca"] = agrupar_raros(df["marca"], MIN_FREQ_MARCA)
    return df.reset_index(drop=True)


def preparar_base(tamanho=config.TAMANHO_AMOSTRA, salvar=True):
    """Executa o pipeline completo e devolve a amostra e um resumo das etapas."""
    bruto = carregar_bruto()
    resumo = {"linhas_brutas": len(bruto)}

    capital = filtrar_capital(bruto)
    resumo["linhas_capital_roubo_furto"] = len(capital)

    boletins, n_ambiguos = consolidar_boletins(capital)
    resumo["boletins_ambiguos_removidos"] = n_ambiguos
    resumo["boletins_unicos"] = len(boletins)

    base = limpar(montar_atributos(boletins))
    resumo["boletins_validos"] = len(base)

    amostra = base.sample(n=min(tamanho, len(base)), random_state=config.SEMENTE).reset_index(drop=True)
    resumo["amostra"] = len(amostra)
    resumo["proporcao_roubo_amostra"] = float(np.round(amostra["roubo"].mean(), 4))

    if salvar:
        config.ARQ_PROCESSADO.parent.mkdir(parents=True, exist_ok=True)
        amostra.to_csv(config.ARQ_PROCESSADO, index=False, encoding="utf-8")
        with open(config.ARQ_RESUMO, "w", encoding="utf-8") as arq:
            json.dump(resumo, arq, ensure_ascii=False, indent=2)
    return amostra, resumo
