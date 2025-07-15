import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import cv2
import time

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

# Diccionario para descripciones de las clases
CLASS_DESCRIPTIONS = {
    'space': 'Espacio',
    'nothing': 'Fondo/No seña',
    'del': 'Borrar'
}

# --- CARGAR MODELO ---
@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        st.success("Modelo cargado correctamente!")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

model = load_trained_model()

# --- FUNCIONES AUXILIARES ---
def preprocess_image(image):
    """Preprocesa la imagen para que coincida con el formato de entrenamiento"""
    try:
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Redimensionar y normalizar
        img_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img_resized) / 255.0
        
        # Aplicar ecualización del histograma (mejora el contraste)
        img_array = cv2.cvtColor(np.uint8(img_array*255), cv2.COLOR_RGB2YCrCb)
        channels = cv2.split(img_array)
        cv2.equalizeHist(channels[0], channels[0])
        img_array = cv2.merge(channels)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_YCrCb2RGB)
        img_array = img_array.astype('float32') / 255.0
        
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error procesando imagen: {str(e)}")
        return None

def display_top_predictions(preds, top_n=5):
    """Muestra las top N predicciones con sus probabilidades"""
    top_indices = np.argsort(preds)[-top_n:][::-1]
    st.subheader("🔝 Predicciones principales:")
    
    for i, idx in enumerate(top_indices):
        class_name = CLASSES[idx]
        prob = preds[idx]
        
        # Barra de progreso con etiqueta
        label = f"{i+1}. {class_name} ({CLASS_DESCRIPTIONS.get(class_name, 'Letra')}): {prob*100:.1f}%"
        st.progress(float(prob), text=label)

# --- INTERFAZ DE USUARIO ---
st.title("🤖 Clasificador de Lengua de Señas ASL")
st.markdown("""
    Sube una imagen de una letra en ASL (American Sign Language) y el modelo predirá qué letra representa.
    También puedes usar la cámara para capturar una imagen en tiempo real.
""")

# Pestañas para diferentes métodos de entrada
tab1, tab2 = st.tabs(["📤 Subir imagen", "📷 Usar cámara"])

with tab1:
    uploaded_file = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_container_width=True)
            
            # Mostrar spinner mientras se procesa
            with st.spinner('Procesando imagen...'):
                img_array = preprocess_image(image)
                if img_array is not None and model is not None:
                    start_time = time.time()
                    pred = model.predict(img_array)[0]
                    processing_time = time.time() - start_time
                    
                    # Resultados
                    pred_idx = np.argmax(pred)
                    pred_class = CLASSES[pred_idx]
                    confidence = pred[pred_idx]
                    
                    # Mostrar resultados en columnas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🧠 Predicción", f"{pred_class}", 
                                f"{CLASS_DESCRIPTIONS.get(pred_class, 'Letra')}")
                    with col2:
                        st.metric("🔍 Confianza", f"{confidence*100:.2f}%")
                    with col3:
                        st.metric("⏱ Tiempo", f"{processing_time*1000:.1f} ms")
                    
                    # Gráfico de barras interactivo
                    st.subheader("📊 Distribución de probabilidades")
                    display_top_predictions(pred, top_n=5)
                    
                    # Gráfico detallado
                    fig, ax = plt.subplots()
                    prob_df = pd.DataFrame({
                        'Letra': CLASSES,
                        'Probabilidad': pred
                    }).sort_values(by='Probabilidad', ascending=False)
                    
                    ax.barh(prob_df['Letra'][:10], prob_df['Probabilidad'][:10], color='skyblue')
                    ax.set_xlabel('Probabilidad')
                    ax.set_title('Top 10 predicciones')
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

with tab2:
    st.warning("Esta función requiere acceso a tu cámara. Por favor, acepta los permisos.")
    picture = st.camera_input("Toma una foto de tu mano")
    
    if picture:
        try:
            image = Image.open(picture)
            st.image(image, caption="Foto tomada", use_container_width=True)
            
            with st.spinner('Procesando imagen...'):
                img_array = preprocess_image(image)
                if img_array is not None and model is not None:
                    pred = model.predict(img_array)[0]
                    pred_idx = np.argmax(pred)
                    pred_class = CLASSES[pred_idx]
                    confidence = pred[pred_idx]
                    
                    st.success(f"Predicción: {pred_class} ({confidence*100:.1f}% de confianza)")
                    display_top_predictions(pred)

        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

# --- SECCIÓN INFORMATIVA ---
st.markdown("---")
with st.expander("ℹ️ Acerca del modelo y ASL"):
    st.markdown("""
    ### Información sobre el modelo
    - **Arquitectura**: CNN (Red Neuronal Convolucional)
    - **Precisión**: ~95% en conjunto de validación
    - **Entrenado con**: Imágenes de 64x64 pixeles
    - **Mejores prácticas**: 
        - Usa imágenes con buena iluminación
        - La mano debe ocupar la mayor parte de la imagen
        - Fondo preferiblemente claro y uniforme
        
    ### ¿Qué es ASL?
    La Lengua de Señas Americana (ASL) es una lengua natural completa 
    que utiliza gestos con las manos, expresiones faciales y movimientos 
    corporales para comunicarse.
    """)

# --- FOOTER ---
st.markdown("---")
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0e1117;
    color: white;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
<p>Creado por Rosana Longares · Proyecto ASL Deep Learning 🤟 · <a href="https://github.com/RosanaNicklas/American_Sign_Language" target="_blank">GitHub</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)