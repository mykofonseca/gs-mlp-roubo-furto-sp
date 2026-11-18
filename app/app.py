"""Aplicação web para testar a MLP de classificação roubo x furto.

Executar a partir da raiz do repositório:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inferencia import ClassificadorOcorrencia  # noqa: E402

ROTULO_PERIODO = {
    "madrugada": "Madrugada (0h às 5h59)",
    "manha": "Manhã (6h às 11h59)",
    "tarde": "Tarde (12h às 17h59)",
    "noite": "Noite (18h às 23h59)",
}
ROTULO_DIA = {
    "segunda": "Segunda-feira", "terca": "Terça-feira", "quarta": "Quarta-feira", "quinta": "Quinta-feira",
    "sexta": "Sexta-feira", "sabado": "Sábado", "domingo": "Domingo",
}


@st.cache_resource
def carregar_modelo():
    return ClassificadorOcorrencia()


def pct(valor, casas=1):
    return f"{valor:.{casas}%}".replace(".", ",")


def numero_dp(nome):
    # Ordena "1º D.P." antes de "10º D.P." em vez da ordem alfabética
    prefixo = nome.split("º")[0]
    return int(prefixo) if prefixo.isdigit() else 999


st.set_page_config(page_title="Roubo ou furto? | MLP SSP-SP", layout="centered")
clf = carregar_modelo()
meta = clf.metadados
opcoes = meta["opcoes"]

st.title("Roubo ou furto?")
st.write(
    "Informe as características de uma ocorrência de celular subtraído na cidade de São Paulo. "
    "A rede neural (MLP) estima se o registro é mais compatível com **roubo** (art. 157) ou "
    "**furto** (art. 155)."
)

col1, col2 = st.columns(2)
with col1:
    delegacia = st.selectbox("Delegacia da área (DP)", sorted(opcoes["delegacia"], key=numero_dp))
    periodo = st.selectbox("Período do dia", opcoes["periodo"], index=3, format_func=ROTULO_PERIODO.get)
    dia = st.selectbox("Dia da semana", opcoes["dia_semana"], index=4, format_func=ROTULO_DIA.get)
    qtd = st.number_input("Celulares levados no boletim", min_value=1, max_value=5, value=1, step=1)
with col2:
    tipos = sorted(opcoes["subtipos_por_tipo"], key=lambda t: (t == "Outros", t))
    tipo_local = st.selectbox("Tipo de local", tipos, index=tipos.index("Via Pública"))
    subtipos = opcoes["subtipos_por_tipo"][tipo_local]
    # Quando existe um subtipo com o mesmo nome do tipo (ex.: Via Pública), ele vem selecionado
    subtipo_local = st.selectbox("Subtipo de local", subtipos,
                                 index=subtipos.index(tipo_local) if tipo_local in subtipos else 0)
    marca = st.selectbox("Marca do aparelho", opcoes["marca"], index=opcoes["marca"].index("Apple"))

if st.button("Classificar ocorrência", type="primary", use_container_width=True):
    registro = {
        "delegacia": delegacia, "periodo": periodo, "dia_semana": dia, "tipo_local": tipo_local,
        "subtipo_local": subtipo_local, "marca": marca, "qtd_celulares": int(qtd),
    }
    resultado = clf.classificar(registro)
    st.subheader(f"Classe estimada: {resultado['classe'].upper()}")
    m1, m2 = st.columns(2)
    m1.metric("Probabilidade de roubo", pct(resultado["prob_roubo"]))
    m2.metric("Probabilidade de furto", pct(resultado["prob_furto"]))
    st.progress(resultado["prob_roubo"], text="Barra preenchida = probabilidade de roubo")
    st.info(
        f"Interpretação: considerando que uma ocorrência aconteceu com essas características, o modelo "
        f"atribui {pct(resultado['prob_roubo'], 0)} de probabilidade à classe roubo e "
        f"{pct(resultado['prob_furto'], 0)} à classe furto. Isso **não** é a chance de um crime acontecer "
        f"nesse local ou horário."
    )

with st.expander("Sobre o modelo"):
    arq = meta["arquitetura"]
    tr = meta["treino"]
    teste = meta["metricas_teste"]
    amostra = f"{meta['fonte']['registros_amostra']:,}".replace(",", ".")
    st.markdown(
        f"""
- **Arquitetura ({meta['modelo']}):** {arq['entradas']} entradas → camadas ocultas
  {' → '.join(map(str, arq['camadas_ocultas']))} ({arq['ativacao']}, dropout {arq['dropout']}) → 1 neurônio sigmoide
- **Treino:** {tr['otimizador']}, taxa {tr['lr']}, weight decay {tr['weight_decay']}, lotes de {tr['batch_size']},
  parada na época {tr['melhor_epoca']}, peso da classe roubo {tr['pos_weight']}
- **Teste (3.000 boletins):** acurácia {pct(teste['acuracia'])}, precision {teste['precision']:.3f},
  recall {teste['recall']:.3f}, F1 {teste['f1']:.3f}
- **Dados:** {meta['fonte']['base']} ({meta['fonte']['orgao']}), amostra de {amostra} boletins da capital. Fonte: {meta['fonte']['url']}
"""
    )

st.caption("Global Solution FIAP – Gustavo Franzoti Gonçalves e Maykon Santana Fonseca")
