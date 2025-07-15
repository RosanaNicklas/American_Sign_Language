import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(
    page_title="ASL Detector",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- CONSTANTES ---
MODEL_PATH = "asl_model_aug.h5"
IMG_SIZE = 64

CLASSES = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing', 'del'
])

# --- CARGAR MODELO ---
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# --- TÍTULO ---
st.title("🤖 Clasificador de Lengua de Señas ASL")
st.markdown("Sube una imagen de una letra en ASL y el modelo predirá qué letra representa.")

# --- SUBIDA DE IMAGEN ---
uploaded_file = st.file_uploader("📸 Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # --- PREPROCESAMIENTO ---
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- PREDICCIÓN ---
    pred = model.predict(img_array)[0]
    pred_idx = np.argmax(pred)
    pred_class = CLASSES[pred_idx]
    confidence = pred[pred_idx]

    # --- DISEÑO EN COLUMNAS ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("🧠 Predicción", f"{pred_class}")
    with col2:
        st.metric("🔍 Confianza", f"{confidence*100:.2f}%")

    # --- GRÁFICO DE PROBABILIDADES ---
    st.markdown("### 📊 Probabilidades por clase")
    prob_df = pd.DataFrame({
        'Letra': CLASSES,
        'Probabilidad': pred
    }).sort_values(by='Probabilidad', ascending=False)

    st.bar_chart(prob_df.set_index("Letra"))

# --- FOOTER ---
st.markdown("---")
st.markdown("Creado por [Rosana Longares] · Proyecto ASL Deep Learning 🤟")
