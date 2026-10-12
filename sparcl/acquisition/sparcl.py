"""SPARCL acquisition (Algorithm 1).

Each round:
  1. For every unlabeled image x, the image uncertainty U(x) and the spatial distribution q(x) of its
     uncertainty over the M = K x L grid bins.
  2. Candidate set: the top-(r * B) images by U(x); U is min-max normalized over the candidates.
  3. Positional-diversity gate: lambda = lambda_max * H(q_bar) / log M, q_bar = mean of q(x) over candidates.
  4. Coverage state H_0 = B / |L+| * sum_{x in L+} g(x), where g(x) is the ground-truth lesion distribution over
     bins of each lesion-bearing labeled image.
  5. Greedy batch: argmax_x U(x) + lambda * sum_m [psi(H_m + q_m(x)) - psi(H_m)], then H += q(x*), psi = sqrt.

Segmentation: H(i) is the binary entropy of the foreground probability, U(x) = mean_i H(i),
h_m(x) = sum_{i in bin m} H(i). Detection: w_b = s_b * H(s_b) for each predicted box b with score s_b,
U(x) = sum_b w_b, h_m(x) = sum_b w_b * |b intersect bin m| / |b|. In both cases q(x) = h(x) / sum_m h_m(x).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

EPS = 1e-12


def bin_index_map(height, width, areas):
    """[H, W] map of grid-bin indices from ``(y0, y1, x0, x1, idx)`` rectangles."""
    bin_ids = np.zeros((height, width), dtype=np.int64)
    for y0, y1, x0, x1, idx in areas:
        bin_ids[y0:y1, x0:x1] = idx
    return bin_ids


def binary_entropy(p):
    p = p.clamp(EPS, 1 - EPS)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def box_bin_overlap(boxes, areas):
    """[N, M] fraction of each box's area that falls inside each bin."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    overlap = np.zeros((len(boxes), len(areas)))
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for y0, y1, x0, x1, idx in areas:
        iw = np.clip(np.minimum(boxes[:, 2], x1) - np.maximum(boxes[:, 0], x0), 0, None)
        ih = np.clip(np.minimum(boxes[:, 3], y1) - np.maximum(boxes[:, 1], y0), 0, None)
        overlap[:, idx] = np.where(box_area > 0, iw * ih / np.maximum(box_area, EPS), 0.0)
    return overlap


def normalize_rows(h):
    total = h.sum(axis=1, keepdims=True)
    return np.where(total > 0, h / np.maximum(total, EPS), 0.0)


@torch.no_grad()
def score_segmentation(model, dataset, indices, areas, device, batch_size=8, num_workers=4):
    """Return ``U [N]`` and ``q [N, M]`` for ``dataset[indices]``."""
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    bin_ids = None
    n_bins = len(areas)
    U, h = [], []
    for images, _ in loader:
        prob = model(images.to(device)).squeeze(1).double()
        if bin_ids is None:
            bin_ids = torch.as_tensor(bin_index_map(*prob.shape[-2:], areas), device=device).flatten()
        entropy = binary_entropy(prob).flatten(1)
        hist = torch.zeros(entropy.shape[0], n_bins, dtype=torch.float64, device=device)
        hist.scatter_add_(1, bin_ids.expand(entropy.shape[0], -1), entropy)
        U.append(entropy.mean(dim=1).cpu())
        h.append(hist.cpu())
    return torch.cat(U).numpy(), normalize_rows(torch.cat(h).numpy())


@torch.no_grad()
def score_detection(model, dataset, indices, areas, device, batch_size=8, collate_fn=None):
    """Return ``U [N]`` and ``q [N, M]`` for ``dataset[indices]`` from the predicted boxes."""
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    U, h = [], []
    for images, _ in loader:
        for pred in model([img.to(device) for img in images]):
            scores = pred['scores'].double().cpu()
            weights = (scores * binary_entropy(scores)).numpy()
            U.append(weights.sum())
            h.append(weights @ box_bin_overlap(pred['boxes'].cpu().numpy(), areas) if len(weights)
                     else np.zeros(len(areas)))
    return np.asarray(U, dtype=np.float64), normalize_rows(np.stack(h))


def mask_lesion_distribution(mask, areas):
    """g(x) from a binary mask [H, W]: fraction of lesion pixels in each bin (None if no lesion)."""
    mask = np.asarray(mask) > 0.5
    total = mask.sum()
    if total == 0:
        return None
    return np.array([mask[y0:y1, x0:x1].sum() for y0, y1, x0, x1, _ in areas], dtype=np.float64) / total


def box_lesion_distribution(boxes, areas):
    """g(x) from ground-truth boxes: fraction of the total box area in each bin (None if no box)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if len(boxes) == 0 or area.sum() <= 0:
        return None
    return (area[:, None] * box_bin_overlap(boxes, areas)).sum(axis=0) / area.sum()


def positional_diversity_gate(q):
    """H(q_bar) / log M with q_bar the mean distribution of the rows of ``q``."""
    n_bins = q.shape[1]
    q_bar = q.mean(axis=0)
    if n_bins < 2 or q_bar.sum() <= 0:
        return 0.0
    q_bar = q_bar / q_bar.sum()
    nz = q_bar[q_bar > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(n_bins)) + 0.0


def select_sparcl(unlabeled, budget, U, q, lesion_distributions, lambda_max=1.0, candidate_ratio=5,
                  psi=np.sqrt):
    """Greedy SPARCL batch.

    Args:
        unlabeled: dataset indices of the unlabeled pool.
        budget: batch size B.
        U, q: uncertainty [N] and spatial distribution [N, M] aligned with ``unlabeled``.
        lesion_distributions: [|L+|, M] ground-truth lesion distributions of the lesion-bearing labeled images.

    Returns:
        (selected dataset indices, diagnostics dict)
    """
    if len(unlabeled) <= budget:
        return list(unlabeled), {}

    n_bins = q.shape[1]
    n_cand = min(len(unlabeled), int(candidate_ratio * budget))
    cand = np.argsort(-U, kind='stable')[:n_cand]
    u = U[cand]
    u = (u - u.min()) / max(u.max() - u.min(), EPS)
    qc = q[cand]

    gate = positional_diversity_gate(qc)
    lam = lambda_max * gate

    g = np.asarray(lesion_distributions, dtype=np.float64).reshape(-1, n_bins)
    coverage = budget * g.mean(axis=0) if len(g) else np.zeros(n_bins)
    h0 = coverage.copy()

    picked = []
    available = np.ones(len(cand), dtype=bool)
    psi_cov = psi(coverage)
    for _ in range(budget):
        gain = u + lam * (psi(coverage[None, :] + qc) - psi_cov[None, :]).sum(axis=1)
        gain[~available] = -np.inf
        best = int(np.argmax(gain))
        picked.append(best)
        available[best] = False
        coverage += qc[best]
        psi_cov = psi(coverage)

    selected = [unlabeled[cand[i]] for i in picked]
    diagnostics = {'gate': gate, 'lambda': lam, 'num_candidates': int(n_cand),
                   'num_lesion_labeled': int(len(g)), 'h0': h0.tolist()}
    return selected, diagnostics
