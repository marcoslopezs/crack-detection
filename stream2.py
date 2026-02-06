import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import os

from torchvision import transforms

# Import local functions
from model_loader import load_faster_rcnn, load_unet_plus_plus
from model_utils import process_detection_output, process_segmentation_output


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Crack Detection System",
    layout="wide",
)

st.title("Crack Detection System")
st.markdown("""
This system uses two **Convolutional Neural Network** models for automatic surface inspection:
- **Faster R-CNN** → crack localization  
- **UNet++** → segmentation and damage quantification
""")


# =========================================================
# Device selection
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Model loading with cache
# =========================================================
@st.cache_resource
def load_models(device: str):
    models = {}

    path_rcnn = "weights/fasterrcnn_final_SGD.pth"
    path_unet = "weights/unetpp_final.pth"

    if not os.path.exists(path_rcnn) or not os.path.exists(path_unet):
        st.error("⚠️ Weight files not found in the 'weights' folder.")
        return None

    try:
        models["faster_rcnn"] = load_faster_rcnn(path_rcnn, device=device)
        models["unetpp"] = load_unet_plus_plus(path_unet, device=device)

        # Explicit evaluation mode
        models["faster_rcnn"].eval()
        models["unetpp"].eval()

        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


models = load_models(DEVICE)

if models is None:
    st.stop()


# =========================================================
# Unified preprocessing
# =========================================================
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Converts a PIL image into a normalized tensor
    compatible with the trained models.
    """
    tensor = image_transform(image).unsqueeze(0)
    return tensor.to(DEVICE)


# =========================================================
# Sidebar interface
# =========================================================
st.sidebar.header("⚙️ Settings")

st.sidebar.info(f"🧠 Device: **{DEVICE.upper()}**")

analysis_mode = st.sidebar.radio(
    "Analysis mode",
    ["Crack Detection", "Segmentation & Measurement"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

# Pixel to mm conversion (optional, only for segmentation)
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Scale Conversion in segmentation mode")
enable_conversion = st.sidebar.checkbox("Enable px → mm conversion", value=False)

px_to_mm = None
if enable_conversion:
    px_to_mm = st.sidebar.number_input(
        "Scale (mm per pixel)",
        min_value=0.001,
        max_value=10.0,
        value=0.1,
        step=0.01,
        format="%.3f",
        help="Enter the scale factor to convert pixels to millimeters"
    )



# =========================================================
# Main area
# =========================================================
uploaded_file = st.file_uploader(
    "📂 Upload an image to analyze",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original image", use_container_width=True)

    if st.button("🚀 Analyze image"):
        with st.spinner("Processing image..."):

            img_tensor = preprocess_image(image)

            # -------------------------------------------------
            # DETECTION MODE
            # -------------------------------------------------
            if analysis_mode == "Crack Detection":
                model = models["faster_rcnn"]

                with torch.no_grad():
                    prediction = model(img_tensor)

                detections = process_detection_output(
                    prediction,
                    threshold=confidence_threshold
                )

                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                if len(detections) > 0:
                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        score = det["score"]

                        cv2.rectangle(
                            img_cv,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            3
                        )

                        cv2.putText(
                            img_cv,
                            f"{score:.2f}",
                            (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                    st.success(f"✅ {len(detections)} crack(s) detected.")
                else:
                    st.info("✅ No cracks detected with this threshold.")

                st.image(
                    cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB),
                    caption="Detection result",
                    use_container_width=True
                )


            # -------------------------------------------------
            # SEGMENTATION MODE
            # -------------------------------------------------
            else:
                model = models["unetpp"]

                original_size = image.size  # (W, H)

                # Controlled resizing for UNet++
                input_resized = torch.nn.functional.interpolate(
                    img_tensor,
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False
                )

                with torch.no_grad():
                    output = model(input_resized)

                metrics, mask_resized = process_segmentation_output(
                    output,
                    original_size,
                    threshold=confidence_threshold
                )

                if metrics["is_positive"]:
                    img_np = np.array(image)
                    overlay = img_np.copy()

                    overlay[mask_resized > 0] = [255, 0, 0]

                    result = cv2.addWeighted(
                        overlay, 0.4,
                        img_np, 0.6,
                        0
                    )

                    st.success("✅ Cracks segmented successfully.")

                    col1, col2 = st.columns(2)
                    col1.metric("Affected area (px)", metrics["area_px"])
                    col2.metric("Estimated extent (px)", metrics["length_px"])

                    # Show mm values if conversion is enabled
                    if px_to_mm is not None:
                        area_mm2 = metrics["area_px"] * (px_to_mm ** 2)
                        length_mm = metrics["length_px"] * px_to_mm
                        
                        col3, col4 = st.columns(2)
                        col3.metric("Affected area (mm²)", f"{area_mm2:.2f}")
                        col4.metric("Estimated extent (mm)", f"{length_mm:.2f}")

                    st.image(
                        result,
                        caption="Segmentation result",
                        use_container_width=True
                    )
                else:
                    st.info("✅ No significant cracks detected.")


else:
    st.info("📌 Upload an image to start the analysis.")
