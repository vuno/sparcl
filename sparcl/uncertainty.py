"""Pool uncertainty used by the baseline strategies.

SPARCL computes its own uncertainty statistics in ``sparcl.acquisition.sparcl``.
"""

import numpy as np
import torch


@torch.no_grad()
def segmentation_uncertainty(model, loader, device, return_predictions=False, return_features=False):
    """Per-image score = mean of the top-10% pixel binary entropies of the foreground probability."""
    model.eval()
    uncertainties, predictions, features = [], [], []
    eps = 1e-8
    for images, _ in loader:
        images = images.to(device)
        if return_features:
            probs, feats = model(images, return_features=True)
            features.append(feats.cpu().numpy())
        else:
            probs = model(images)
        probs = probs.squeeze(1).cpu().numpy()

        p = np.clip(probs, eps, 1 - eps)
        p = np.nan_to_num(p, nan=0.5, posinf=1 - eps, neginf=eps)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        for ent in entropy.reshape(len(images), -1):
            ent = np.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)
            top_k = max(1, int(0.1 * len(ent)))
            score = float(np.mean(np.sort(ent)[-top_k:]))
            uncertainties.append(score if np.isfinite(score) else float(np.mean(ent)))
        if return_predictions:
            predictions.append(probs)

    result = {'uncertainties': np.array(uncertainties)}
    if return_predictions:
        result['predictions'] = np.concatenate(predictions, axis=0)
    if return_features:
        result['features'] = np.concatenate(features, axis=0)
    return result


@torch.no_grad()
def detection_uncertainty(model, loader, device, return_predictions=False, return_features=False):
    """Per-image score = 0.6 * (1 - mean box confidence) + 0.4 * detection-count term (1.0 if no box)."""
    model.eval()
    uncertainties, predictions, features, sizes = [], [], [], []
    for images, _ in loader:
        images = [img.to(device) for img in images]
        sizes.extend(tuple(img.shape[-2:]) for img in images)
        if return_features:
            preds, feats = model(images, return_features=True)
            features.append(feats.cpu().numpy())
        else:
            preds = model(images)

        for pred in preds:
            scores = pred['scores'].cpu().numpy()
            if len(scores) == 0:
                uncertainties.append(1.0)
                continue
            n = len(scores)
            count_term = 0.2 if n <= 3 else min(0.5 + (n - 3) * 0.1, 1.0)
            uncertainties.append(0.6 * (1.0 - np.mean(scores)) + 0.4 * count_term)
        if return_predictions:
            predictions.extend({k: v.cpu() for k, v in p.items()} for p in preds)

    result = {'uncertainties': np.array(uncertainties)}
    if return_predictions:
        result['predictions'] = predictions
        result['probs'] = np.stack([box_score_map(p, size) for p, size in zip(predictions, sizes)])
    if return_features:
        result['features'] = np.concatenate(features, axis=0)
    return result


def box_score_map(pred, size):
    """Dense map holding, at every pixel, the highest score of the predicted boxes covering it."""
    h, w = size
    score_map = np.zeros((h, w), dtype=np.float32)
    for box, score in zip(pred['boxes'].numpy(), pred['scores'].numpy()):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            score_map[y1:y2, x1:x2] = np.maximum(score_map[y1:y2, x1:x2], score)
    return score_map
