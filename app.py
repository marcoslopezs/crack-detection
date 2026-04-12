from __future__ import annotations

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from inference_utils import SEGMENTATION_IMAGE_SIZE, image_bytes_to_tensor, load_project_models, resolve_device
from model_utils import process_detection_output, process_segmentation_output

app = FastAPI(
    title="Crack Detection API",
    description="FastAPI backend for crack localisation and crack segmentation.",
    version="1.0.0",
)

models = {}
DEVICE = resolve_device()


def ensure_models_loaded():
    if models:
        return

    models.update(load_project_models(DEVICE))


@app.on_event("startup")
def startup_event():
    ensure_models_loaded()


def get_model(name: str):
    model = models.get(name)
    if model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    return model


def decode_uploaded_image(image_bytes: bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    return decoded


@app.post("/predict/detection")
async def predict_detection(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    image_bytes = await file.read()
    image_tensor, _, _ = image_bytes_to_tensor(image_bytes, device=DEVICE)

    with torch.no_grad():
        prediction = get_model("faster_rcnn")(image_tensor)

    detections = process_detection_output(prediction, threshold)
    is_positive = len(detections) > 0

    return {
        "filename": file.filename,
        "is_positive": is_positive,
        "detection_count": len(detections),
        "detections": detections,
        "status": "positive" if is_positive else "negative",
    }


@app.post("/predict/detection/image")
async def predict_detection_image(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    image_bytes = await file.read()
    image_tensor, _, _ = image_bytes_to_tensor(image_bytes, device=DEVICE)

    with torch.no_grad():
        prediction = get_model("faster_rcnn")(image_tensor)

    detections = process_detection_output(prediction, threshold)
    image = decode_uploaded_image(image_bytes)

    if detections:
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            score = detection["score"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                image,
                f"{score:.2f}",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
        label = f"positive ({len(detections)} crack{'s' if len(detections) != 1 else ''})"
        label_color = (0, 0, 255)
    else:
        label = "negative"
        label_color = (0, 160, 0)

    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
    _, encoded = cv2.imencode(".jpg", image)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.post("/predict/segmentation")
async def predict_segmentation(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    scale_param: float | None = Query(default=None, gt=0.0),
):
    image_bytes = await file.read()
    image_tensor, original_size, _ = image_bytes_to_tensor(
        image_bytes,
        device=DEVICE,
        target_size=SEGMENTATION_IMAGE_SIZE,
    )

    with torch.no_grad():
        output = get_model("unetpp")(image_tensor)

    metrics, _ = process_segmentation_output(output, original_size, threshold, scale_param)
    return {"filename": file.filename, **metrics}


@app.post("/predict/segmentation/image")
async def predict_segmentation_image(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    image_bytes = await file.read()
    image_tensor, original_size, _ = image_bytes_to_tensor(
        image_bytes,
        device=DEVICE,
        target_size=SEGMENTATION_IMAGE_SIZE,
    )

    with torch.no_grad():
        output = get_model("unetpp")(image_tensor)

    metrics, mask_resized = process_segmentation_output(output, original_size, threshold)
    image = decode_uploaded_image(image_bytes)

    if metrics["is_positive"]:
        overlay = image.copy()
        overlay[mask_resized > 0] = [0, 0, 255]
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        label = "positive"
        label_color = (0, 0, 255)
    else:
        label = "negative"
        label_color = (0, 160, 0)

    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
    _, encoded = cv2.imencode(".jpg", image)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
