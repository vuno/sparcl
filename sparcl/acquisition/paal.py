"""PAAL: predictive accuracy-based active learning (Shi et al., IJCAI 2024), 'ap' query mode.

An accuracy predictor regresses the segmenter's per-image Dice from [image ; foreground probability]. It is
fine-tuned on the labeled set after every round and persists across rounds; the query takes the images with
the lowest predicted Dice.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..models import AccuracyPredictor
from .simple import select_uncertainty

EPOCHS_PER_ROUND = 8
LR = 1e-3
EPS = 1e-6


def _dice_per_image(prob, mask, threshold=0.5):
    p = (prob > threshold).float().flatten(1)
    g = (mask > 0.5).float().flatten(1)
    return (2 * (p * g).sum(1) + EPS) / (p.sum(1) + g.sum(1) + EPS)


class PAAL:
    def __init__(self, device):
        self.device = device
        self.predictor = None
        self.optimizer = None
        self.trained = False

    def fit(self, seg_model, labeled_dataset, batch_size=8):
        """Regress the segmenter's own per-image Dice on the labeled set."""
        if len(labeled_dataset) == 0:
            return
        if self.predictor is None:
            self.predictor = AccuracyPredictor(in_channels=4).to(self.device)
            self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=LR, weight_decay=1e-4)
        loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        seg_model.eval()
        self.predictor.train()
        mse = nn.MSELoss()
        for _ in range(EPOCHS_PER_ROUND):
            for images, masks in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                with torch.no_grad():
                    prob = seg_model(images)
                    target = _dice_per_image(prob, masks).unsqueeze(1)
                loss = mse(self.predictor(torch.cat([images, prob], dim=1)), target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.trained = True

    @torch.no_grad()
    def predict(self, dataset, indices, probs, batch_size=8):
        """Predicted Dice of ``dataset[indices]`` given their foreground probabilities ``probs[indices]``."""
        self.predictor.eval()
        scores = []
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
        offset = 0
        for images, _ in loader:
            prob = torch.as_tensor(np.asarray(probs)[indices[offset:offset + len(images)]],
                                   dtype=torch.float32).unsqueeze(1)
            offset += len(images)
            inp = torch.cat([images.float(), prob], dim=1).to(self.device)
            scores.extend(self.predictor(inp).squeeze(1).cpu().numpy().tolist())
        return np.array(scores)

    def select(self, unlabeled, budget, dataset, probs, uncertainties, seed=None):
        if len(unlabeled) <= budget:
            return list(unlabeled)
        if not self.trained or probs is None:
            return select_uncertainty(unlabeled, budget, uncertainties, seed)
        scores = self.predict(dataset, list(unlabeled), probs)
        return [unlabeled[i] for i in np.argsort(scores, kind='stable')[:budget]]
