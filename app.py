import streamlit as st
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator
import tempfile
import os

st.title("Detección de Rostros con Audio 🤖")

st.write("📸 Sube una imagen o toma una foto para detectar rostros y escuchar el resultado en diferentes idiomas.")

# --- Opción: subir imagen o usar cámara ---
option = st.radio("Elige cómo quieres ingresar la imagen:", ("📁 Subir imagen", "🎥 Usar cámara"))

if option == "📁 Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen:", type=["jpg", "png", "jpeg"])
elif option == "🎥 Usar cámara":
    uploaded_file = st.camera_input("Toma una foto")

# --- Configuración de idioma y acento ---
with st.sidebar:
    st.subheader("Configuración de idioma y acento")

    out_lang = st.selectbox(
        "Idioma de salida",
        ("Español", "Inglés", "Francés", "Alemán", "Italiano", "Japonés"),
    )

    lang_codes = {
        "Español": "es",
        "Inglés": "en",
        "Francés": "fr",
        "Alemán": "de",
        "Italiano": "it",
        "Japonés": "ja",
    }

    accent = st.selectbox(
        "Acento de voz",
        ("Default", "India", "United Kingdom", "United States", "Australia", "South Africa"),
    )

    tld_map = {
        "Default": "com",
        "India": "co.in",
        "United Kingdom": "co.uk",
        "United States": "com",
        "Australia": "com.au",
        "South Africa": "co.za",
    }
    tld = tld_map[accent]

# --- Detector de rostros ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Procesamiento de imagen ---
if uploaded_file is not None:
    # Convertir a formato que OpenCV pueda leer
    img = np.array(Image.open(uploaded_file).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Dibujar rectángulos en los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    st.image(img, caption="Resultado de detección", use_container_width=True)

    # Crear mensaje con el resultado
    num_faces = len(faces)
    if num_faces == 0:
        message = "No se detectaron rostros en la imagen."
    elif num_faces == 1:
        message = "Se detectó 1 rostro en la imagen."
    else:
        message = f"Se detectaron {num_faces} rostros en la imagen."

    st.subheader("Resultado:")
    st.write(message)

    # --- Generar audio del resultado ---
    if st.button("🔊 Reproducir audio del resultado"):
        translated_message = GoogleTranslator(source="es", target=lang_codes[out_lang]).translate(message)
        tts = gTTS(translated_message, lang=lang_codes[out_lang], tld=tld, slow=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            audio_path = temp_audio.name

        st.audio(audio_path, format="audio/mp3")
        st.success("✅ Audio generado con éxito")

 
    
    
