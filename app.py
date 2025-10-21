import streamlit as st
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator
import tempfile
import os

st.title("Detección de Rostros con Audio 🤖")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen para analizar:", type=["jpg", "png", "jpeg"])

# Sidebar de configuración
with st.sidebar:
    st.subheader("Configuración de idioma y acento")

    out_lang = st.selectbox(
        "Idioma de salida",
        ("Español", "Inglés", "Francés", "Alemán", "Italiano", "Japonés"),
    )

    # Mapeo de idiomas a códigos ISO
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

    # Mapeo de acentos
    tld_map = {
        "Default": "com",
        "India": "co.in",
        "United Kingdom": "co.uk",
        "United States": "com",
        "Australia": "com.au",
        "South Africa": "co.za",
    }
    tld = tld_map[accent]

# Cargar el detector de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if uploaded_file is not None:
    # Leer imagen
    img = np.array(Image.open(uploaded_file).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Dibujar rectángulos en los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar imagen con detección
    st.image(img, caption="Resultado de detección", use_container_width=True)

    # Crear mensaje
    num_faces = len(faces)
    if num_faces == 0:
        message = "No se detectaron rostros en la imagen."
    elif num_faces == 1:
        message = "Se detectó 1 rostro en la imagen."
    else:
        message = f"Se detectaron {num_faces} rostros en la imagen."

    st.subheader("Resultado:")
    st.write(message)

    # Traducir y convertir a voz
    if st.button("🔊 Reproducir audio del resultado"):
        translated_message = GoogleTranslator(source="es", target=lang_codes[out_lang]).translate(message)
        tts = gTTS(translated_message, lang=lang_codes[out_lang], tld=tld, slow=False)

        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            audio_path = temp_audio.name

        st.audio(audio_path, format="audio/mp3")
        st.success("✅ Audio generado con éxito")



 
    
    
