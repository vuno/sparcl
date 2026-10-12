import os

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class DetectionModel(nn.Module):
    """COCO-pretrained Faster R-CNN (ResNet-50 FPN) with a lesion-vs-background box predictor."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None, return_features=False):
        if isinstance(images, torch.Tensor):
            images = list(images) if images.dim() == 4 else [images]

        if self.training and targets is not None:
            loss_dict = self.model(images, targets)
            return {'loss': sum(loss_dict.values()), **loss_dict}

        predictions = self.model(images)
        if not return_features:
            return predictions

        with torch.no_grad():
            features = []
            for img in images:
                feat = self.model.backbone(img.unsqueeze(0))['3']
                features.append(feat.mean(dim=(2, 3)).squeeze())
        return predictions, torch.stack(features)


def create_detection_model(args, device, checkpoint_path=None):
    model = DetectionModel().to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state.get('model_state_dict', state), strict=False)
    return model
