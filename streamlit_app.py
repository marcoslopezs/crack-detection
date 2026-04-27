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
SCALE_DEFAULT = 0.1
SCALE_STEP = 0.01


@st.cache_resource
def load_models(device_name: str):
    return load_project_models(device_name)


def render_html_block(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def format_scale_value(value: float) -> str:
    return f"{value:.3f}"


def parse_scale_value(raw_value: str) -> float | None:
    normalized = raw_value.strip().replace(",", ".")
    if not normalized:
        return None

    try:
        parsed = float(normalized)
    except ValueError:
        return None

    if parsed <= 0:
        return None

    return parsed


def update_scale_value(delta: float):
    current_value = parse_scale_value(st.session_state.get("scale_input_value", ""))
    if current_value is None:
        current_value = SCALE_DEFAULT

    updated_value = max(0.001, min(10.0, current_value + delta))
    st.session_state["scale_input_value"] = format_scale_value(updated_value)


def normalize_scale_input():
    current_value = parse_scale_value(st.session_state.get("scale_input_value", ""))
    if current_value is not None:
        st.session_state["scale_input_value"] = format_scale_value(current_value)


def render_summary(title: str, status_text: str, tone: str, body: str):
    render_html_block(
        f"""
        <div class="summary-band">
            <span class="status-pill {tone}">{status_text}</span>
            <h2>{title}</h2>
            <p>{body}</p>
        </div>
        """
    )


st.markdown(
    """
    <style>
    :root {
        --page: #edf2f4;
        --page-line: rgba(17, 24, 39, 0.06);
        --surface: #ffffff;
        --surface-alt: #f7fafb;
        --ink: #17212b;
        --muted: #495866;
        --accent: #b64926;
        --accent-deep: #91361b;
        --teal: #296b73;
        --line: #d7e0e5;
        --ok-bg: #e8f4ed;
        --ok-ink: #246746;
        --warn-bg: #fbe9e3;
        --warn-ink: #8f391e;
    }

    html, body, [class*="css"] {
        color: var(--ink);
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(var(--page-line) 1px, transparent 1px),
            linear-gradient(90deg, var(--page-line) 1px, transparent 1px),
            linear-gradient(180deg, #f7fbfc 0%, var(--page) 100%);
        background-size: 28px 28px, 28px 28px, auto;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 251, 252, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #f3f6f7;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink);
    }

    [data-testid="stSidebar"] code {
        background: rgba(41, 107, 115, 0.10);
        color: var(--teal);
        padding: 0.08rem 0.35rem;
        border: 1px solid rgba(41, 107, 115, 0.18);
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        background: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-baseweb="input"] {
        background: #ffffff !important;
        border: 1px solid #9db0ba !important;
    }

    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background: #ffffff !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        min-height: 2.5rem;
        padding: 0.2rem 0.2rem;
        background: #000000;
        color: #ffffff;
        border: 1px solid #000000;
        border-radius: 6px;
        font-size: 1.1rem;
        font-weight: 700;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #000000;
        color: #ffffff;
        border-color: #000000;
    }

    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span {
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
        margin: 0 !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        font-family: "Segoe UI", "Arial", sans-serif !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] button[aria-label*="increment"],
    [data-testid="stSidebar"] button[aria-label*="decrement"],
    [data-testid="stSidebar"] button[title*="increment"],
    [data-testid="stSidebar"] button[title*="decrement"] {
        color: #ffffff !important;
        fill: #ffffff !important;
        background: #000000 !important;
        border-color: #000000 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"] *,
    [data-testid="stSidebar"] button[aria-label*="increment"] *,
    [data-testid="stSidebar"] button[aria-label*="decrement"] *,
    [data-testid="stSidebar"] button[title*="increment"] *,
    [data-testid="stSidebar"] button[title*="decrement"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"] svg,
    [data-testid="stSidebar"] button[aria-label*="increment"] svg,
    [data-testid="stSidebar"] button[aria-label*="decrement"] svg,
    [data-testid="stSidebar"] button[title*="increment"] svg,
    [data-testid="stSidebar"] button[title*="decrement"] svg,
    [data-testid="stSidebar"] button[kind="secondary"] svg path,
    [data-testid="stSidebar"] button[aria-label*="increment"] svg path,
    [data-testid="stSidebar"] button[aria-label*="decrement"] svg path,
    [data-testid="stSidebar"] button[title*="increment"] svg path,
    [data-testid="stSidebar"] button[title*="decrement"] svg path {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="input"] button {
        min-width: 2.25rem !important;
        background: #000000 !important;
        border-left: 1px solid #000000 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="input"] button:hover {
        background: #000000 !important;
    }

    .scale-input-label {
        margin: 0.35rem 0 0.3rem;
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 500;
    }

    .scale-help {
        margin-top: 0.35rem;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.35;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    label,
    .stRadio label,
    .stCheckbox label {
        color: var(--ink);
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px dashed #9db0ba;
        border-radius: 10px;
    }

    .hero {
        padding: 1.8rem 0 1.2rem;
        border-bottom: 2px solid var(--line);
        margin-bottom: 1.1rem;
    }

    .hero-kicker {
        margin: 0 0 0.45rem;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--teal);
        font-weight: 700;
    }

    .hero h1 {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 3rem;
        line-height: 1.02;
        color: var(--ink);
    }

    .hero p {
        max-width: 52rem;
        margin: 0.85rem 0 0;
        font-size: 1.05rem;
        line-height: 1.65;
        color: var(--muted);
    }

    .hero-meta {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .hero-meta span {
        display: inline-flex;
        align-items: center;
        min-height: 2.1rem;
        padding: 0.35rem 0.8rem;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.88);
        font-size: 0.92rem;
        color: var(--ink);
    }

    .surface {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid var(--line);
        padding: 1rem 1.05rem;
        margin-bottom: 1rem;
    }

    .surface h3 {
        margin: 0 0 0.45rem;
        font-size: 1rem;
        color: var(--ink);
    }

    .surface p {
        margin: 0;
        font-size: 0.97rem;
        line-height: 1.55;
        color: var(--muted);
    }

    .section-label {
        margin: 1rem 0 0.45rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--teal);
        font-weight: 700;
    }

    .run-card dl {
        margin: 0;
    }

    .run-card dt {
        margin: 0 0 0.16rem;
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .run-card dd {
        margin: 0 0 0.8rem;
        color: var(--ink);
        font-size: 1rem;
        line-height: 1.35;
    }

    .summary-band {
        margin: 1rem 0 0.9rem;
        padding: 1.05rem 1.15rem;
        border-left: 5px solid var(--accent);
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 1px 0 rgba(23, 33, 43, 0.04);
    }

    .summary-band h2 {
        margin: 0;
        font-size: 1.35rem;
        color: var(--ink);
    }

    .summary-band p {
        margin: 0.55rem 0 0;
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.98rem;
    }

    .status-pill {
        display: inline-block;
        margin-bottom: 0.55rem;
        padding: 0.34rem 0.72rem;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .status-positive {
        background: var(--warn-bg);
        color: var(--warn-ink);
    }

    .status-negative {
        background: var(--ok-bg);
        color: var(--ok-ink);
    }

    .stButton > button {
        min-height: 2.8rem;
        padding: 0.6rem 1.3rem;
        background: var(--accent);
        color: white;
        border: 1px solid var(--accent);
        border-radius: 6px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: var(--accent-deep);
        color: white;
        border-color: var(--accent-deep);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid var(--line);
        padding: 0.85rem 0.95rem;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink);
    }

    @media (max-width: 900px) {
        .hero h1 {
            font-size: 2.25rem;
        }

        .hero p {
            font-size: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_html_block(
    """
    <section class="hero">
        <p class="hero-kicker">Computer Vision Project</p>
        <h1>Crack Detection</h1>
        <p>
            You can run a quick localisation test with the detection model <b>Faster R-CNN</b> or switch to the segmentation model <b>UNet++</b> for a clearer crack
            mask and rough area and length estimates.
        </p>
        <div class="hero-meta">
            <span>Detection and segmentation</span>
            <span>CPU and CUDA compatible</span>
            <span>Prepared for Hugging Face Spaces</span>
        </div>
    </section>
    """
)

try:
    models = load_models(str(DEVICE))
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.sidebar.markdown("## Run settings")
st.sidebar.caption(f"Device: {DEVICE.type.upper()}")

analysis_mode = st.sidebar.radio(
    "Analysis mode",
    ["Crack detection", "Segmentation and measurement"],
    captions=[
        "Bounding boxes and confidence scores.",
        "Mask overlay with area and length metrics.",
    ],
)

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.05,
    help="Higher values make the model more selective.",
)

scale_param = None
if analysis_mode == "Segmentation and measurement":
    use_scale = st.sidebar.checkbox("Convert pixels to millimetres", value=False)
    if use_scale:
        if "scale_input_value" not in st.session_state:
            st.session_state["scale_input_value"] = format_scale_value(SCALE_DEFAULT)

        render_html_block('<p class="scale-input-label">Millimetres per pixel</p>')
        scale_input_col, scale_minus_col, scale_plus_col = st.sidebar.columns([5, 1, 1])

        with scale_input_col:
            st.text_input(
                "Millimetres per pixel value",
                key="scale_input_value",
                label_visibility="collapsed",
                help="Use a point as decimal separator, for example 0.100.",
                on_change=normalize_scale_input,
            )

        with scale_minus_col:
            st.button(
                "−",
                key="scale_decrement",
                use_container_width=True,
                on_click=update_scale_value,
                args=(-SCALE_STEP,),
            )

        with scale_plus_col:
            st.button(
                "＋",
                key="scale_increment",
                use_container_width=True,
                on_click=update_scale_value,
                args=(SCALE_STEP,),
            )

        scale_param = parse_scale_value(st.session_state.get("scale_input_value", ""))
        if scale_param is None:
            st.sidebar.caption("Enter a value between 0.001 and 10.0 using a decimal point.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Image advice**

    - Use close, sharp images.
    - Avoid strong shadows over the crack.
    - For segmentation, keep the damaged area clearly visible.
    """
)


uploaded_file = st.file_uploader(
    "Upload an inspection image",
    type=["jpg", "jpeg", "png"],
    help="Accepted formats: JPG, JPEG and PNG.",
)

if uploaded_file is None:
    render_html_block(
        """
        <div class="surface">
            <h3>Start with one image</h3>
            <p>
                Upload a close-up image of the surface and then run either detection or segmentation. The
                interface is tuned for quick comparison between the original image and the model output.
            </p>
        </div>
        """
    )
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
file_size_kb = len(uploaded_file.getvalue()) / 1024

preview_col, run_col = st.columns([1.45, 0.8])

with preview_col:
    render_html_block('<p class="section-label">Input image</p>')
    st.image(image, caption="Original image", use_container_width=True)

with run_col:
    render_html_block('<p class="section-label">Current run</p>')
    render_html_block(
        f"""
        <div class="surface run-card">
            <h3>Session details</h3>
            <dl>
                <dt>Mode</dt>
                <dd>{analysis_mode}</dd>
                <dt>Threshold</dt>
                <dd>{confidence_threshold:.2f}</dd>
                <dt>Image file</dt>
                <dd>{uploaded_file.name}<br>{file_size_kb:.1f} KB</dd>
                <dt>Scale conversion</dt>
                <dd>{"Enabled" if scale_param is not None else "Disabled"}</dd>
            </dl>
        </div>
        """
    )

if not st.button("Run analysis", type="primary"):
    st.stop()

with st.spinner("Running inference..."):
    if analysis_mode == "Crack detection":
        image_tensor = image_to_tensor(image, device=DEVICE)
        with torch.no_grad():
            prediction = models["faster_rcnn"](image_tensor)

        detections = process_detection_output(prediction, threshold=confidence_threshold)
        result_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        top_score = max((item["score"] for item in detections), default=0.0)

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

        render_summary(
            title="Detection summary",
            status_text="Crack detected" if detections else "No crack detected",
            tone="status-positive" if detections else "status-negative",
            body=(
                f"The model found {len(detections)} candidate crack(s) above the selected threshold. "
                f"The highest confidence score was {top_score:.2f}."
                if detections
                else f"No candidate cracks passed the current threshold of {confidence_threshold:.2f}."
            ),
        )

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Detections", len(detections))
        metric_2.metric("Top confidence", f"{top_score:.2f}")
        metric_3.metric("Threshold", f"{confidence_threshold:.2f}")

        result_left, result_right = st.columns(2)
        result_left.image(image, caption="Original image", use_container_width=True)
        result_right.image(
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
            body = (
                f"A crack mask was detected above the current threshold of {confidence_threshold:.2f}. "
                "These measurements are image-based estimates unless the scale is calibrated."
            )
        else:
            body = f"No segmented crack region passed the current threshold of {confidence_threshold:.2f}."

        render_summary(
            title="Segmentation summary",
            status_text="Segmented crack area" if metrics["is_positive"] else "No segmented crack area",
            tone="status-positive" if metrics["is_positive"] else "status-negative",
            body=body,
        )

        metric_1, metric_2 = st.columns(2)
        metric_1.metric("Affected area (px)", metrics["area_px"])
        metric_2.metric("Estimated length (px)", metrics["length_px"])

        if scale_param is not None:
            metric_3, metric_4 = st.columns(2)
            metric_3.metric("Affected area (mm^2)", f"{metrics['area_mm']:.2f}")
            metric_4.metric("Estimated length (mm)", f"{metrics['length_mm']:.2f}")

        result_left, result_right = st.columns(2)
        result_left.image(image, caption="Original image", use_container_width=True)
        result_right.image(result_image, caption="Segmentation overlay", use_container_width=True)
