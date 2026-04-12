from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from inference_utils import SEGMENTATION_IMAGE_SIZE, image_to_tensor, load_project_models, resolve_device
from model_utils import process_detection_output, process_segmentation_output

st.set_page_config(page_title="Crack Detection", layout="wide")

DEVICE = resolve_device()


@st.cache_resource
def load_models(device_name: str):
    return load_project_models(device_name)


st.markdown(
    """
    <style>
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(245, 243, 239, 1), rgba(232, 236, 241, 0.85));
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.6rem 0 0;
        max-width: 48rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Crack Detection Demo</h1>
        <p>
            I built this project to turn the models from my bachelor thesis into a small inspection tool.
            It supports two workflows: object detection for quick localisation and segmentation for a more
            detailed estimate of damaged area and crack length.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Running on {DEVICE.type.upper()}. The first prediction can take a few seconds because the weights are loaded on demand."
)

try:
    models = load_models(str(DEVICE))
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.sidebar.header("Settings")
analysis_mode = st.sidebar.radio(
    "Analysis mode",
    ["Crack detection", "Segmentation and measurement"],
)
confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

scale_param = None
if analysis_mode == "Segmentation and measurement":
    use_scale = st.sidebar.checkbox("Convert pixels to millimetres", value=False)
    if use_scale:
        scale_param = st.sidebar.number_input(
            "Millimetres per pixel",
            min_value=0.001,
            max_value=10.0,
            value=0.1,
            step=0.01,
            format="%.3f",
        )

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image to run the demo.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
left_col, right_col = st.columns(2)
left_col.image(image, caption="Original image", use_container_width=True)

if not st.button("Run analysis", type="primary"):
    st.stop()

with st.spinner("Running inference..."):
    if analysis_mode == "Crack detection":
        image_tensor = image_to_tensor(image, device=DEVICE)
        with torch.no_grad():
            prediction = models["faster_rcnn"](image_tensor)

        detections = process_detection_output(prediction, threshold=confidence_threshold)
        result_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            score = detection["score"]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                result_image,
                f"{score:.2f}",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        if detections:
            st.success(f"{len(detections)} crack detection(s) above the selected threshold.")
        else:
            st.info("No detections above the selected threshold.")

        right_col.image(
            cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
            caption="Detection overlay",
            use_container_width=True,
        )

    else:
        image_tensor = image_to_tensor(image, device=DEVICE, target_size=SEGMENTATION_IMAGE_SIZE)
        with torch.no_grad():
            output = models["unetpp"](image_tensor)

        metrics, mask_resized = process_segmentation_output(
            output,
            image.size,
            threshold=confidence_threshold,
            scale_param=scale_param,
        )

        result_image = np.array(image)
        if metrics["is_positive"]:
            overlay = result_image.copy()
            overlay[mask_resized > 0] = [255, 0, 0]
            result_image = cv2.addWeighted(overlay, 0.35, result_image, 0.65, 0)
            st.success("Segmentation completed.")
        else:
            st.info("No segmented crack area above the selected threshold.")

        right_col.image(result_image, caption="Segmentation overlay", use_container_width=True)

        metrics_col_1, metrics_col_2 = st.columns(2)
        metrics_col_1.metric("Affected area (px)", metrics["area_px"])
        metrics_col_2.metric("Estimated length (px)", metrics["length_px"])

        if scale_param is not None:
            metrics_col_3, metrics_col_4 = st.columns(2)
            metrics_col_3.metric("Affected area (mm^2)", f"{metrics['area_mm']:.2f}")
            metrics_col_4.metric("Estimated length (mm)", f"{metrics['length_mm']:.2f}")
