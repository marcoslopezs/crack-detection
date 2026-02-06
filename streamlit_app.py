import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import io
import os

# Importar funciones locales
from model_loader import load_faster_rcnn, load_unet_plus_plus
from model_utils import process_detection_output, process_segmentation_output

# Configuración de página
st.set_page_config(page_title="Detección de Fisuras - TFG", layout="wide", page_icon="🏗️")

st.title("🏗️ Sistema de Detección de Fisuras")
st.markdown("""
Este sistema utiliza dos modelos de redes neuronales convolucionales para inspeccionar imágenes de superficies (concreto, pavimento, paredes) en busca de fisuras.
Combina detección de objetos (**Faster R-CNN**) para localizar daños y segmentación semántica (**UNet++**) para cuantificarlos.
""")

# --- Carga de Modelos con Caché ---
@st.cache_resource
def load_models():
    device = "cpu" # Forzar CPU en Spaces gratuito
    models = {}
    
    # Paths relativos
    path_rcnn = "weights/fasterrcnn_final_SGD.pth"
    path_unet = "weights/unetpp_final.pth"
    
    if not os.path.exists(path_rcnn) or not os.path.exists(path_unet):
        st.error("⚠️ No se encontraron los archivos de pesos en la carpeta 'weights'.")
        return None

    try:
        models["faster_rcnn"] = load_faster_rcnn(path_rcnn, device=device)
        models["unetpp"] = load_unet_plus_plus(path_unet, device=device)
        return models
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None

models = load_models()

if models is None:
    st.stop()

# --- Interfaz Lateral ---
st.sidebar.header("⚙️ Configuración")
analysis_mode = st.sidebar.radio(
    "Modo de Análisis",
    ["Detección de Fisuras", "Segmentación y Medición"]
)

confidence_threshold = st.sidebar.slider("Umbral de Confianza", 0.0, 1.0, 0.5, 0.05)

# --- Área Principal ---
uploaded_file = st.file_uploader("📂 Sube una imagen para analizar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Cargar y mostrar imagen original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen Original", use_column_width=True)
    
    # Botón Análisis
    if st.button("🚀 Analizar Imagen"):
        with st.spinner("Procesando..."):
            
            # Preprocesamiento común
            img_array = np.array(image)
            img_tensor = torch.tensor(img_array / 255.0).permute(2, 0, 1).float().unsqueeze(0)
            
            # --- MODO DETECCIÓN ---
            if analysis_mode == "Detección de Fisuras":
                model = models["faster_rcnn"]
                with torch.no_grad():
                    prediction = model(img_tensor)
                
                detections = process_detection_output(prediction, threshold=confidence_threshold)
                
                # Dibujar
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                if len(detections) > 0:
                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        score = det["score"]
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(img_cv, f"{score:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    st.success(f"✅ Se detectaron {len(detections)} fisuras.")
                else:
                    st.info("✅ No se detectaron fisuras con este umbral.")
                
                # Resultado
                st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Resultado de Detección", use_column_width=True)
                
            # --- MODO SEGMENTACIÓN ---
            else:
                model = models["unetpp"]
                # Para UNet++ podemos redimensionar si es necesario, pero aquí usaremos size original
                # Nota: Tu código original redimensionaba a 512 en app.py, lo replicamos si quieres consistencia
                # o lo dejamos full size. Lo haremos full size pero controlando memoria.
                
                # Si es muy grande, resize. 
                original_size = image.size # W, H
                
                # Inferencia
                # OJO: UNet necesita dimensiones divisibles por 32 a veces, 
                # pero tu entrenamiento parece usar size 512.
                # Replicando app.py: resize a 512 para inferencia, resize back para mask.
                input_tensor_resized = torch.nn.functional.interpolate(img_tensor, size=(512, 512), mode='bilinear', align_corners=False)
                
                with torch.no_grad():
                    output = model(input_tensor_resized)
                
                # Procesar
                metrics, mask_resized = process_segmentation_output(output, original_size, threshold=confidence_threshold)
                
                if metrics["is_positive"]:
                    # Overlay Rojo
                    img_cv = np.array(image)
                    overlay = img_cv.copy()
                    
                    # Máscara roja
                    overlay[mask_resized > 0] = [255, 0, 0] 
                    
                    # Blend
                    img_result = cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0)
                    
                    st.success("✅ Fisuras segmentadas.")
                    col1, col2 = st.columns(2)
                    col1.metric("Área (px)", metrics["area_px"])
                    col2.metric("Longitud Estimada (px)", metrics["length_px"])
                    
                    st.image(img_result, caption="Segmentación", use_column_width=True)
                else:
                    st.info("✅ No hay fisuras significativas.")
else:
    st.info("Esperando imagen para comenzar...")
