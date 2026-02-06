import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import segmentation_models_pytorch as smp

def load_faster_rcnn(weights_path, device="cpu"):
    # Según tu TFG: Faster R-CNN con backbone ResNet50 [cite: 3227, 3272]
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    # Ajuste a 2 clases (fondo y fisura) como hiciste en Kaggle [cite: 3273]
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def load_unet_plus_plus(weights_path, device="cpu"):
    # Según tu TFG: UNet++ con ResNet34 [cite: 3421]
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model