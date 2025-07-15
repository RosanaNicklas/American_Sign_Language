import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import os

# --- FUNCIONES AUXILIARES PARA INFORMACIÓN DE LETRAS ---
def get_hand_position(letter):
    positions = {
        'A': "Puño cerrado con el pulgar al lado",
        'B': "Mano plana con dedos juntos y pulgar doblado",
        'C': "Mano curvada en forma de C, dedos ligeramente separados",
        'D': "Dedo índice apuntando hacia arriba, otros dedos cerrados",
        'E': "Dobla todos los dedos hacia la palma con el pulgar sobre ellos",
        'F': "Pulgar e índice formando un círculo, otros dedos extendidos",
        'G': "Dedo índice apuntando hacia un lado, otros dedos cerrados",
        'H': "Dedos índice y medio extendidos y juntos, otros cerrados",
        'I': "Dedo meñique extendido, otros dedos cerrados",
        'J': "Dedo meñique extendido, hacer movimiento de J en el aire",
        'K': "Dedos índice y medio extendidos y separados en V, pulgar extendido",
        'L': "Dedos índice y pulgar extendidos formando una L",
        'M': "Puño cerrado con pulgar debajo de los otros dedos",
        'N': "Puño cerrado con pulgar entre los dedos índice y medio",
        'O': "Todos los dedos curvados para formar un círculo",
        'P': "Dedos índice y medio extendidos hacia abajo, pulgar cruzado",
        'Q': "Dedo índice apuntando hacia abajo, pulgar extendido",
        'R': "Dedos índice y medio cruzados",
        'S': "Puño cerrado con pulgar sobre los dedos",
        'T': "Puño cerrado con pulgar entre índice y medio",
        'U': "Dedos índice y medio extendidos y juntos hacia arriba",
        'V': "Dedos índice y medio extendidos y separados en V",
        'W': "Dedos índice, medio y anular extendidos y separados",
        'X': "Dedo índice doblado en forma de gancho",
        'Y': "Dedo meñique y pulgar extendidos, otros dedos cerrados",
        'Z': "Dedo índice dibujando una Z en el aire"
    }
    return positions.get(letter, "Información no disponible")

def get_hand_movement(letter):
    movements = {
        'A': "Ninguno, posición estática",
        'B': "Pequeño movimiento de lado a lado",
        'C': "Ninguno, posición estática",
        'D': "Ninguno, posición estática",
        'E': "Ninguno, posición estática",
        'F': "Ninguno, posición estática",
        'G': "Ninguno, posición estática",
        'H': "Ninguno, posición estática",
        'I': "Ninguno, posición estática",
        'J': "Mover la mano trazando una J en el aire",
        'K': "Ninguno, posición estática",
        'L': "Ninguno, posición estática",
        'M': "Ninguno, posición estática",
        'N': "Ninguno, posición estática",
        'O': "Ninguno, posición estática",
        'P': "Ninguno, posición estática",
        'Q': "Ninguno, posición estática",
        'R': "Ninguno, posición estática",
        'S': "Ninguno, posición estática",
        'T': "Ninguno, posición estática",
        'U': "Ninguno, posición estática",
        'V': "Ninguno, posición estática",
        'W': "Ninguno, posición estática",
        'X': "Ninguno, posición estática",
        'Y': "Ninguno, posición estática",
        'Z': "Mover la mano trazando una Z en el aire"
    }
    return movements.get(letter, "Sin movimiento específico")

def get_example_word(letter):
    examples = {
        'A': "Apple (manzana)",
        'B': "Ball (pelota)",
        'C': "Cat (gato)",
        'D': "Dog (perro)",
        'E': "Elephant (elefante)",
        'F': "Fish (pez)",
        'G': "Girl (niña)",
        'H': "House (casa)",
        'I': "Ice (hielo)",
        'J': "Jump (saltar)",
        'K': "King (rey)",
        'L': "Love (amor)",
        'M': "Mother (madre)",
        'N': "No (no)",
        'O': "Orange (naranja)",
        'P': "Play (jugar)",
        'Q': "Queen (reina)",
        'R': "Rain (lluvia)",
        'S': "Sun (sol)",
        'T': "Tree (árbol)",
        'U': "Up (arriba)",
        'V': "Van (furgoneta)",
        'W': "Water (agua)",
        'X': "Xylophone (xilófono)",
        'Y': "Yellow (amarillo)",
        'Z': "Zebra (cebra)"
    }
    return examples.get(letter, "Ejemplo no disponible")

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(
    page_title="ASL Detector Pro",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTES ---
MODEL_PATH = "asl_model_aug.h5"
IMG_SIZE = 64

CLASSES = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing', 'del'
])

CLASS_DESCRIPTIONS = {
    'space': 'Espacio',
    'nothing': 'Fondo/No seña',
    'del': 'Borrar'
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- CARGAR MODELO ---
@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

model = load_trained_model()

# --- FUNCIONES AUXILIARES ---
def preprocess_image(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        img_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img_resized) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error procesando imagen: {str(e)}")
        return None

def predict_image(image):
    if model is None:
        return None, 0
    img_array = preprocess_image(image)
    if img_array is None:
        return None, 0
    pred = model.predict(img_array)[0]
    pred_idx = np.argmax(pred)
    pred_class = CLASSES[pred_idx]
    confidence = pred[pred_idx]
    return pred_class, confidence

def update_phrase_history(pred_class):
    if 'phrase_history' not in st.session_state:
        st.session_state.phrase_history = []
    if pred_class == 'del' and st.session_state.phrase_history:
        st.session_state.phrase_history.pop()
    elif pred_class == 'space':
        st.session_state.phrase_history.append(' ')
    elif pred_class not in ['nothing', 'del']:
        st.session_state.phrase_history.append(pred_class)

# --- INTERFAZ PRINCIPAL ---
st.title("🤟 ASL Detector Pro")
st.markdown("""
    Una aplicación completa para detectar y traducir el lenguaje de señas americano (ASL).
    Utiliza inteligencia artificial para reconocer letras en tiempo real y formar palabras.
""")

with st.sidebar:
    st.header("Opciones")
    app_mode = st.radio("Modo de aplicación:", 
                       ["Reconocimiento", "Aprende ASL", "Historial"])
    
    if app_mode == "Reconocimiento":
        input_mode = st.radio("Entrada:", ["Imagen", "Cámara", "Video en tiempo real"])
    
    st.markdown("---")
    st.markdown("### Historial rápido")
    if 'phrase_history' in st.session_state and st.session_state.phrase_history:
        st.write("".join(st.session_state.phrase_history))
    else:
        st.write("No hay historial aún")

# --- MODO RECONOCIMIENTO ---
if app_mode == "Reconocimiento":
    if input_mode == "Imagen":
        st.header("Sube una imagen con señal ASL")
        uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_container_width=True)
            pred_class, confidence = predict_image(image)
            if pred_class is not None:
                st.write(f"Predicción: **{pred_class}** con confianza {confidence:.2f}")
            else:
                st.warning("No se pudo realizar la predicción.")
    
    elif input_mode == "Cámara":
        st.header("Usa tu cámara para tomar una foto")
        picture = st.camera_input("Toma una foto")
        if picture:
            image = Image.open(picture)
            st.image(image, caption="Foto tomada", use_column_width=True)
            pred_class, confidence = predict_image(image)
            if pred_class is not None:
                st.write(f"Predicción: **{pred_class}** con confianza {confidence:.2f}")
            else:
                st.warning("No se pudo realizar la predicción.")

    elif input_mode == "Video en tiempo real":
        st.header("Detección en video en tiempo real")
        st.write("Usa tu cámara para detectar signos ASL mientras hablas con las manos.")
        
        if 'phrase_history' not in st.session_state:
            st.session_state.phrase_history = []
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
        if 'last_prediction_time' not in st.session_state:
            st.session_state.last_prediction_time = 0
        
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_class, confidence = predict_image(img_rgb)
            
            current_time = time.time()
            # Evitar añadir repetidos muy seguidos (1 segundo)
            if pred_class and confidence > 0.7:
                if pred_class != st.session_state.last_prediction or (current_time - st.session_state.last_prediction_time) > 1:
                    st.session_state.last_prediction = pred_class
                    st.session_state.last_prediction_time = current_time
                    update_phrase_history(pred_class)
            
            # Mostrar la predicción en el video
            if pred_class:
                cv2.putText(img, f'{pred_class} ({confidence*100:.1f}%)', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="asl-video",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

# --- MODO APRENDE ASL ---
elif app_mode == "Aprende ASL":
    st.header("Aprende las letras del alfabeto ASL")
    col1, col2 = st.columns(2)

    with col1:
        letter = st.selectbox("Selecciona una letra para aprender", sorted(CLASSES[:-3]))  # Excluye space, nothing, del
        st.subheader(f"Letra: {letter}")
        # Carpeta donde tienes las letras con ejemplos
        IMAGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "asl_alphabet_test")

        # Ruta a la subcarpeta de la letra elegida
        letter_dir = os.path.join(IMAGE_BASE_DIR, letter)

        # Coge la primera imagen .jpg o .png que aparezca
        img_path = None
        if os.path.isdir(letter_dir):
            for fname in sorted(os.listdir(letter_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(letter_dir, fname)
                    break

        if img_path and os.path.exists(img_path):
            st.image(img_path,
                caption=f"Ejemplo de la letra {letter}",
                use_container_width=True)
        else:
            st.warning("No se encontró ninguna imagen de ejemplo para esta letra.")


    with col2:
        st.subheader("Detalles de la seña")
        st.markdown(f"**Posición de la mano:** {get_hand_position(letter)}")
        st.markdown(f"**Movimiento:** {get_hand_movement(letter)}")
        st.markdown(f"**Ejemplo de palabra:** {get_example_word(letter)}")

# --- MODO HISTORIAL ---
elif app_mode == "Historial":
    st.header("Historial de letras y frases detectadas")
    if 'phrase_history' in st.session_state and st.session_state.phrase_history:
        phrase = "".join(st.session_state.phrase_history)
        st.markdown(f"### Frase detectada:")
        st.write(phrase)
        if st.button("Borrar historial"):
            st.session_state.phrase_history = []
            st.success("Historial borrado.")
    else:
        st.write("No hay historial aún.")



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