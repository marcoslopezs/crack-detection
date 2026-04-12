try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import torch
from skimage.morphology import skeletonize


def process_segmentation_output(prediction_tensor, original_size, threshold=0.5, scale_param=None):
    if cv2 is None:
        raise RuntimeError("OpenCV is required to resize segmentation masks.")

    prob_mask = torch.sigmoid(prediction_tensor).squeeze().cpu().numpy()
    binary_mask = (prob_mask >= threshold).astype(np.uint8)
    binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)

    area_px = int(binary_mask_resized.sum())
    skeleton = skeletonize(binary_mask_resized > 0)
    length_px = int(skeleton.sum())
    is_positive = area_px > 0

    results = {
        "is_positive": is_positive,
        "area_px": area_px,
        "length_px": length_px,
        "status": "positive" if is_positive else "negative",
    }

    if scale_param is not None:
        results["area_mm"] = area_px * (scale_param ** 2)
        results["length_mm"] = length_px * scale_param

    return results, binary_mask_resized


def process_detection_output(prediction, threshold=0.5):
    boxes = prediction[0]["boxes"].cpu().numpy().tolist()
    scores = prediction[0]["scores"].cpu().numpy().tolist()

    valid_detections = []

    for box, score in zip(boxes, scores):
        if score > threshold:
            valid_detections.append({
                "box": [int(b) for b in box],
                "score": float(score),
            })

    valid_detections.sort(key=lambda x: x["score"], reverse=True)
    return valid_detections
