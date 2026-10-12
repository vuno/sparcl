import os

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

ENCODERS = {'efficientnet-b4': 'efficientnet-b4', 'resnet50': 'resnet50'}


class SegmentationModel(nn.Module):
    """U-Net with an ImageNet-pretrained encoder, scSE decoder attention and a sigmoid output."""

    def __init__(self, encoder='efficientnet-b4'):
        super().__init__()
        self.model = smp.Unet(encoder_name=ENCODERS[encoder], encoder_weights='imagenet',
                              decoder_attention_type='scse', in_channels=3, classes=1,
                              activation='sigmoid')

    def forward(self, x, return_features=False):
        mask = self.model(x)
        if not return_features:
            return mask
        features = self.model.encoder(x)[-1].mean(dim=(2, 3))
        return mask, features


class LossPredictionModule(nn.Module):
    """Loss prediction head of Learning Loss (Yoo & Kweon, CVPR 2019): GAP -> FC -> ReLU per level, concat -> FC."""

    def __init__(self, feature_dims, hidden_dim=128):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True))
            for dim in feature_dims
        ])
        self.fc = nn.Linear(hidden_dim * len(feature_dims), 1)

    def forward(self, feature_maps):
        return self.fc(torch.cat([branch(f) for branch, f in zip(self.branches, feature_maps)], dim=1))


class SegmentationModelWithLossPrediction(nn.Module):
    """Segmentation model plus a loss prediction head on the last three encoder stages."""

    def __init__(self, base_model, hidden_dim=128):
        super().__init__()
        self.base_model = base_model
        feature_dims = base_model.model.encoder.out_channels[-3:]
        self.loss_prediction_module = LossPredictionModule(feature_dims, hidden_dim)

    def forward(self, x, return_loss_prediction=False, return_features=False):
        if return_features:
            mask, features = self.base_model(x, return_features=True)
        else:
            mask = self.base_model(x)
        if not return_loss_prediction:
            return (mask, features) if return_features else mask

        feature_maps = self.base_model.model.encoder(x)[-3:]
        predicted_loss = self.loss_prediction_module(feature_maps)
        if return_features:
            return mask, predicted_loss, features
        return mask, predicted_loss


def create_segmentation_model(args, device, checkpoint_path=None):
    model = SegmentationModel(args.encoder)
    if args.mode == 'loss_prediction':
        model = SegmentationModelWithLossPrediction(model)
    model = model.to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state.get('model_state_dict', state), strict=False)
    return model
