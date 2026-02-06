from fastapi import FastAPI, UploadFile, File, Query
from PIL import Image
import io
import torch
import numpy as np
import cv2
from fastapi.responses import Response


# Importamos tus módulos propios
from model_loader import load_faster_rcnn, load_unet_plus_plus
from model_utils import process_detection_output, process_segmentation_output

app = FastAPI(title="API Detección de Fisuras")

# Variables globales para los modelos
models = {}
DEVICE = "cpu" # En producción gratis usamos CPU

# --- Evento de arranque: Cargar modelos al iniciar ---
@app.on_event("startup")
def startup_event():
    print("Cargando modelos... esto puede tardar un poco.")
    
    # Asegurar que el directorio de weights existe o manejar error
    import os
    if not os.path.exists("weights"):
        print("ADVERTENCIA: No se encontró el directorio 'weights'. La API fallará al predecir.")
        return

    try:
        models["faster_rcnn"] = load_faster_rcnn("weights/fasterrcnn_final_SGD.pth", device=DEVICE)
        models["unetpp"] = load_unet_plus_plus("weights/unetpp_final.pth", device=DEVICE)
        print("¡Modelos cargados y listos!")
    except Exception as e:
        print(f"ERROR: Falló la carga de modelos: {e}")

# --- Funciones auxiliares de imagen ---
def load_image_into_tensor(image_bytes, target_size=None):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size # (width, height)
    
    if target_size:
        image = image.resize((target_size, target_size))
        
    img_array = np.array(image)
    # Normalizar / 255.0 y trasponer a (C, H, W)
    img_tensor = torch.tensor(img_array / 255.0).permute(2, 0, 1).float().unsqueeze(0)
    return img_tensor.to(DEVICE), original_size

# --- ENDPOINT 1: Detección (Faster R-CNN) ---
@app.post("/predict/detection")
async def predict_detection(file: UploadFile = File(...), threshold: float = 0.5):
    """
    Devuelve las cajas (bounding boxes) donde hay fisuras.
    """
    image_bytes = await file.read()
    # Faster R-CNN acepta tamaños variables, pero transformamos a tensor
    img_tensor, _ = load_image_into_tensor(image_bytes)
    
    with torch.no_grad():
        prediction = models["faster_rcnn"](img_tensor)
        
    detections = process_detection_output(prediction, threshold)
    
    # Añadimos la clasificación binaria
    is_positive = len(detections) > 0
    
    return {
        "filename": file.filename,
        "is_positive": is_positive,
        "detection_count": len(detections),
        "detections": detections,
        "status": "Positivo" if is_positive else "Negativo"
    }

# --- ENDPOINT 2: Detección con imágen (Faster R-CNN) ---
@app.post("/predict/detection/image")
async def predict_detection_image(file: UploadFile = File(...), threshold: float = 0.5):
    image_bytes = await file.read()
    img_tensor, _ = load_image_into_tensor(image_bytes)
    
    with torch.no_grad():
        prediction = models["faster_rcnn"](img_tensor)
    
    # 1. Filtrar cajas con el threshold del usuario
    detections = process_detection_output(prediction, threshold)
    is_positive = len(detections) > 0
    
    # 2. Preparar imagen
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
  # 3. Dibujar
    if is_positive:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            # Caja roja
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Texto con confianza
            cv2.putText(img_cv, f"{score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # --- Lógica Singular/Plural ---
        count = len(detections)
        texto_etiqueta = "fisura" if count == 1 else "fisuras"
        
        # Etiqueta general con la corrección gramatical
        cv2.putText(img_cv, f"POSITIVO ({count} {texto_etiqueta})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    else:
        # Etiqueta Verde si no hay nada
        cv2.putText(img_cv, "NEGATIVO", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', img_cv)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

# --- ENDPOINT 3: Segmentación y Cuantificación (UNet++) ---
@app.post("/predict/segmentation")
async def predict_segmentation(
    file: UploadFile = File(...), 
    threshold: float = 0.5,
    scale_param: float = Query(None)
):
    image_bytes = await file.read()
    img_tensor, original_size = load_image_into_tensor(image_bytes, target_size=512)
    
    with torch.no_grad():
        output = models["unetpp"](img_tensor)
        
    # Llamamos a la utilidad pasándole el threshold
    metrics, _ = process_segmentation_output(output, original_size, threshold, scale_param)
    
    return {"filename": file.filename, **metrics}

# --- ENDPOINT 4: Segmentación y Cuantificación con imágen (UNet++) ---
@app.post("/predict/segmentation/image")
async def predict_segmentation_image(file: UploadFile = File(...), threshold: float = 0.5):
    image_bytes = await file.read()
    img_tensor, original_size = load_image_into_tensor(image_bytes, target_size=512)
    
    with torch.no_grad():
        output = models["unetpp"](img_tensor)
    
    # 1. Usamos la MISMA lógica que el endpoint JSON
    metrics, mask_resized = process_segmentation_output(output, original_size, threshold)
    is_positive = metrics["is_positive"]

    # 2. Preparar imagen original con OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. Dibujar Overlay solo si es positivo
    if is_positive:
        overlay = img_cv.copy()
        # Pintar rojo (BGR: 0, 0, 255)
        overlay[mask_resized > 0] = [0, 0, 255]
        cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0, img_cv)
        
        # Etiqueta visual
        cv2.putText(img_cv, f"POSITIVO", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Etiqueta visual Verde
        cv2.putText(img_cv, "NEGATIVO (Sin fisuras)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', img_cv)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")