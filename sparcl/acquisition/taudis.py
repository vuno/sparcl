"""TAUDIS: two-step uncertainty and diversity sampling (Yu et al., ICCV 2023, arXiv:2309.16139).

Instances are connected components of the predicted foreground (or of the box score map for detection).
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom

from .simple import select_uncertainty

EPS = 1e-8


def _instances(prob_map, min_area=10):
    labeled, n = ndimage.label(prob_map > 0.5)
    instances = []
    for k in range(1, n + 1):
        mask = (labeled == k).astype(np.float32)
        if mask.sum() >= min_area:
            instances.append(mask)
    return instances


def _instance_feature(image_feature, mask):
    """Image embedding concatenated with the instance's normalized center, size and bounding box."""
    h, w = mask.shape
    ys, xs = np.where(mask > 0.5)
    if len(ys) > 0:
        cy, cx = ys.mean() / h, xs.mean() / w
        y_min, y_max, x_min, x_max = ys.min() / h, ys.max() / h, xs.min() / w, xs.max() / w
    else:
        cy = cx = y_min = y_max = x_min = x_max = 0.5
    spatial = np.array([cy, cx, mask.sum() / (h * w), y_min, y_max, x_min, x_max,
                        y_max - y_min, x_max - x_min], dtype=np.float32)
    return np.concatenate([image_feature, spatial])


def _max_k_cover(similarity, k, threshold):
    """Greedy maximum k-set cover: pick rows covering the most still-uncovered columns (similarity >= threshold)."""
    covers = similarity >= threshold
    n_rows = covers.shape[0]
    k = min(k, n_rows)
    selected, covered, remaining = [], np.zeros(covers.shape[1], dtype=bool), set(range(n_rows))
    for _ in range(k):
        if covered.all():
            if remaining:
                selected.append(remaining.pop())
            break
        best_row, best_gain = -1, -1
        for i in list(remaining):
            gain = (covers[i] & ~covered).sum()
            if gain > best_gain:
                best_row, best_gain = i, gain
        if best_row == -1:
            if remaining:
                selected.append(remaining.pop())
            break
        selected.append(best_row)
        remaining.discard(best_row)
        covered |= covers[best_row]
    return selected


def select_taudis(unlabeled, budget, uncertainties, features, probs, alpha=2.5, beta=1.5, sigma=0.8, seed=None):
    if len(unlabeled) < budget:
        return list(unlabeled)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    if features is None or probs is None:
        return select_uncertainty(unlabeled, budget, uncertainties, seed)

    features, probs = np.asarray(features), np.asarray(probs)
    inst_image, inst_feature, inst_uncertainty = [], [], []
    for idx in unlabeled:
        prob = probs[idx]
        if uncertainties is not None:
            p = np.clip(prob, EPS, 1 - EPS)
            entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        else:
            entropy = np.full_like(prob, 0.5)
        for mask in _instances(prob):
            inst_image.append(idx)
            inst_feature.append(_instance_feature(features[idx], mask))
            inst_uncertainty.append(float((entropy * mask).sum() / (mask.sum() + EPS)))

    if not inst_image:
        return rng.choice(unlabeled, size=budget, replace=False).tolist()

    top = np.argsort(np.array(inst_uncertainty))[-int(alpha * budget):][::-1].tolist()
    if not top:
        return rng.choice(unlabeled, size=budget, replace=False).tolist()

    feats = np.array(inst_feature)
    feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-10)
    similarity = np.clip(feats[top] @ feats.T, -1.0, 1.0)
    chosen = _max_k_cover(similarity, min(int(beta * budget), len(top)), sigma)

    votes = {}
    for row in chosen:
        image = inst_image[top[row]]
        votes[image] = votes.get(image, 0) + 1
    selected = [image for image, _ in sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:budget]]

    if len(selected) < budget:
        rest = [idx for idx in unlabeled if idx not in selected]
        if rest:
            selected += rng.choice(rest, size=min(budget - len(selected), len(rest)), replace=False).tolist()
    return selected
