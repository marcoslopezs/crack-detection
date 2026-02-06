import streamlit as st
import requests
from PIL import Image
import io

# Configuración de la página
st.set_page_config(page_title="Detector de Fisuras (Cliente API)", layout="wide", page_icon="🔌")

st.title("🔌 Cliente de Detección (Conecta a API Local)")
st.markdown("""
Esta interfaz se comunica con el servidor FastAPI (`app.py`) corriendo en tu máquina.
Asegúrate de ejecutar primero `uvicorn app:app --reload` en otra terminal.
""")

# Configuración de la URL de la API (Localmente suele ser esta)
API_URL = "http://127.0.0.1:8000"

# 1. Selector de modo (Flags/Botones)
st.sidebar.header("Configuración del Análisis")
option = st.sidebar.radio(
    "Selecciona el Endpoint:",
    (
        "Detección (JSON)", 
        "Detección (Imagen)", 
        "Segmentación (JSON)", 
        "Segmentación (Imagen)"
    )
)

# Mapeo de opciones a los endpoints de tu app.py
endpoint_map = {
    "Detección (JSON)": "/predict/detection",
    "Detección (Imagen)": "/predict/detection/image",
    "Segmentación (JSON)": "/predict/segmentation",
    "Segmentación (Imagen)": "/predict/segmentation/image"
}

# 2. Subida de archivos
uploaded_file = st.file_uploader("Arrastra tu imagen aquí...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen original
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen original subida", width=400)
    
    if st.button("Ejecutar Análisis"):
        with st.spinner('Procesando...'):
            # Preparar el archivo para enviarlo a FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            target_url = f"{API_URL}{endpoint_map[option]}"
            
            try:
                response = requests.post(target_url, files=files)
                
                if response.status_code == 200:
                    # 3. Mostrar resultados según el tipo de respuesta
                    if "image" in option:
                        # La respuesta es un chorro de bytes de imagen
                        result_img = Image.open(io.BytesIO(response.content))
                        st.subheader("Resultado Visual")
                        st.image(result_img, caption=f"Resultado de {option}")
                    else:
                        # La respuesta es un JSON con métricas
                        st.subheader("Métricas")
                        st.json(response.json())
                else:
                    st.error(f"Error en la API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"No se pudo conectar con la API. ¿Está encendida? Error: {e}")

else:
    st.info("Por favor, sube una imagen para comenzar.")