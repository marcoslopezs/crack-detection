from __future__ import annotations

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import segmentation_models_pytorch as smp


def _load_state_dict(weights_path: str, device: str | torch.device):
    return torch.load(weights_path, map_location=device, weights_only=True)


def load_faster_rcnn(weights_path: str, device: str | torch.device = "cpu"):
    device = torch.device(device)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    model.load_state_dict(_load_state_dict(weights_path, device))
    model.to(device)
    model.eval()
    return model


def load_unet_plus_plus(weights_path: str, device: str | torch.device = "cpu"):
    device = torch.device(device)

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(_load_state_dict(weights_path, device))
    model.to(device)
    model.eval()
    return model
