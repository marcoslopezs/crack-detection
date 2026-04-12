from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model_loader import load_faster_rcnn, load_unet_plus_plus

DETECTION_WEIGHTS = Path("weights/fasterrcnn_final_SGD.pth")
SEGMENTATION_WEIGHTS = Path("weights/unetpp_final.pth")
SEGMENTATION_IMAGE_SIZE = 512


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    forced_device = os.getenv("CRACK_DETECTION_DEVICE")
    if forced_device:
        return torch.device(forced_device)

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def image_to_tensor(image: Image.Image, device: str | torch.device = "cpu", target_size: int | None = None) -> torch.Tensor:
    working_image = image.convert("RGB")
    if target_size is not None:
        working_image = working_image.resize((target_size, target_size))

    image_array = np.asarray(working_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def image_bytes_to_tensor(
    image_bytes: bytes,
    device: str | torch.device = "cpu",
    target_size: int | None = None,
):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image_to_tensor(image, device=device, target_size=target_size), image.size, image


def load_project_models(device: str | torch.device):
    missing_files = [
        str(path)
        for path in (DETECTION_WEIGHTS, SEGMENTATION_WEIGHTS)
        if not path.exists()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing weight files: {missing}")

    return {
        "faster_rcnn": load_faster_rcnn(str(DETECTION_WEIGHTS), device=device),
        "unetpp": load_unet_plus_plus(str(SEGMENTATION_WEIGHTS), device=device),
    }
