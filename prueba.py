import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Riesgo de inundación - Santander", layout="wide")

FEATURES = ["lluvia_mm", "altitud_m", "dist_rio_km"]


# ---------------------------------------------------------------
# PASO 1. Datos de municipios
# OJO: lluvia y distancia a ríos son valores ILUSTRATIVOS.
# En el hackathon reemplázalos por datos reales (IDEAM, DANE, SGC).
# ---------------------------------------------------------------
@st.cache_data
def cargar_municipios():
    filas = [
        # municipio, lat, lon, altitud_m, poblacion, lluvia_mm_anual, dist_rio_km
        ("Bucaramanga", 7.1254, -73.1198, 959, 580000, 130, 2.0),
        ("Floridablanca", 7.0622, -73.0864, 925, 270000, 1400, 1.5),
        ("Girón", 7.0683, -73.1697, 780, 200000, 1200, 1.0),
        ("Piedecuesta", 6.9875, -73.0535, 1005, 170000, 1500, 1.0),
        ("Barrancabermeja", 7.0653, -73.8547, 75, 190000, 2800, 0.5),
        ("San Gil", 6.5559, -73.1347, 1109, 55000, 1400, 0.5),
        ("Socorro", 6.4638, -73.2592, 1230, 31000, 1700, 2.0),
        ("Málaga", 6.7019, -72.7311, 2200, 20000, 1200, 3.0),
        ("Puerto Wilches", 7.3475, -73.8992, 80, 31000, 2900, 0.5),
        ("Sabana de Torres", 7.3908, -73.4989, 125, 22000, 2600, 3.0),
        ("San Vicente de Chucurí", 6.8825, -73.4098, 1100, 33000, 2800, 4.0),
        ("Barbosa", 5.9331, -73.6144, 1600, 33000, 2400, 1.0),
        ("Vélez", 6.0133, -73.6742, 2000, 20000, 2300, 2.0),
        ("Cimitarra", 6.3167, -73.9500, 160, 33000, 3000, 1.0),
        ("Lebrija", 7.1167, -73.2167, 1000, 42000, 1100, 4.0),
        ("Zapatoca", 6.8167, -73.2667, 1720, 10000, 1600, 6.0),
        ("Charalá", 6.2833, -73.1500, 1300, 12000, 2000, 2.0),
        ("Oiba", 6.2625, -73.2989, 1500, 11000, 2200, 2.0),
        ("Curití", 6.6033, -73.0695, 1000, 13000, 1500, 1.0),
        ("Barichara", 6.6358, -73.2232, 1336, 8000, 1500, 3.0),
    ]
    cols = ["municipio", "lat", "lon", "altitud_m", "poblacion", "lluvia_mm", "dist_rio_km"]
    return pd.DataFrame(filas, columns=cols)


# ---------------------------------------------------------------
# PASO 2. Histórico de eventos (simulado)
# Con datos reales sería: un registro por municipio/mes con
# lluvia, y una columna "hubo_inundacion" (0/1) de la UNGRD.
# ---------------------------------------------------------------
@st.cache_data
def generar_historico(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    base = cargar_municipios().sample(n, replace=True, random_state=seed).reset_index(drop=True)
    base["lluvia_mm"] = base["lluvia_mm"] * rng.uniform(0.6, 1.5, n)
    score = (
        0.002 * (base["lluvia_mm"] - 1800)
        - 0.002 * (base["altitud_m"] - 900)
        - 0.4 * (base["dist_rio_km"] - 2)
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-score))
    base["inundacion"] = (rng.random(n) < prob).astype(int)
    return base


# ---------------------------------------------------------------
# PASO 3. Entrenar el modelo
# ---------------------------------------------------------------
@st.cache_resource
def entrenar():
    hist = generar_historico()
    X_train, X_test, y_train, y_test = train_test_split(
        hist[FEATURES], hist["inundacion"], test_size=0.25, random_state=42
    )
    modelo = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    return modelo, acc


# ---------------------------------------------------------------
# PASO 4. Interfaz
# ---------------------------------------------------------------
st.title("🌧️ Mapa de riesgo de inundación en Santander")
st.caption("Prototipo UTSmart IA Challenge · datos ilustrativos, reemplazar por IDEAM / DANE / SGC")

modelo, acc = entrenar()

st.sidebar.header("Escenario")
factor = st.sidebar.slider("Cambio en la lluvia (%)", -30, 60, 0, step=5)
st.sidebar.caption("Cuánto aumenta o disminuye la lluvia respecto al promedio. "
                    "Positivo = año lluvioso, negativo = año seco.")
peso_pob = st.sidebar.slider("Peso de la población en la prioridad", 0.0, 1.0, 0.5, step=0.1)
st.sidebar.metric("Presicion del modelo", f"{acc:.1%}")

# Predicción por municipio con el escenario elegido
df = cargar_municipios().copy()
df["lluvia_mm"] = df["lluvia_mm"] * (1 + factor / 100)
df["prob_inundacion"] = modelo.predict_proba(df[FEATURES])[:, 1]

pob_norm = np.log(df["poblacion"]) / np.log(df["poblacion"]).max()
df["prioridad"] = df["prob_inundacion"] * ((1 - peso_pob) + peso_pob * pob_norm)
df["nivel"] = pd.cut(
    df["prob_inundacion"], [0, 0.33, 0.66, 1.0], labels=["Bajo", "Medio", "Alto"], include_lowest=True
)

# Colores y tamaños para el mapa
df["color"] = df["prob_inundacion"].apply(
    lambda p: [int(255 * p), int(200 * (1 - p)), 40, 190]
)
df["radio"] = np.sqrt(df["poblacion"]) * 40

tab_mapa, tab_rank, tab_vars = st.tabs(["🗺️ Mapa", "📋 Ranking", "📊 Variables"])

with tab_mapa:
    capa = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radio",
        pickable=True,
    )
    vista = pdk.ViewState(latitude=6.9, longitude=-73.3, zoom=7.2)
    st.pydeck_chart(
        pdk.Deck(
            layers=[capa],
            initial_view_state=vista,
            tooltip={"text": "{municipio}\nProbabilidad: {prob_inundacion}\nNivel: {nivel}"},
        )
    )
    st.caption("Color: probabilidad de inundación (verde → rojo). Tamaño: población.")

with tab_rank:
    ranking = df.sort_values("prioridad", ascending=False)[
        ["municipio", "nivel", "prob_inundacion", "prioridad", "poblacion"]
    ].reset_index(drop=True)
    ranking.index += 1
    st.dataframe(ranking, use_container_width=True)
    st.download_button("Descargar CSV", ranking.to_csv().encode("utf-8"), "ranking_riesgo.csv")

with tab_vars:
    imp = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values()
    st.subheader("¿Qué variables pesan más en el modelo?")
    st.bar_chart(imp)
    st.write(
        "Esta gráfica le explica al jurado **por qué** el modelo dice lo que dice. "
        "Es clave para que una alcaldía confíe en el resultado."
    )