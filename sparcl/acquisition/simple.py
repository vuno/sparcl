"""Random, uncertainty and feature-diversity baselines.

All strategies take ``unlabeled`` (dataset indices still in the pool, in pool order) and per-sample arrays
indexed by dataset index, and return a list of dataset indices.
"""

import numpy as np


def _rng(seed):
    return np.random.RandomState(seed) if seed is not None else np.random


def select_random(unlabeled, budget, seed=None):
    if len(unlabeled) < budget:
        return list(unlabeled)
    return _rng(seed).choice(unlabeled, size=budget, replace=False).tolist()


def select_uncertainty(unlabeled, budget, uncertainties, seed=None):
    """Top-``budget`` images by uncertainty (random on the first round)."""
    if len(unlabeled) < budget:
        return list(unlabeled)
    if uncertainties is None:
        return select_random(unlabeled, budget, seed)
    return sorted(unlabeled, key=lambda i: uncertainties[i], reverse=True)[:budget]


def _round_robin(groups, budget, limit):
    selected, g, empty_streak = [], 0, 0
    while len(selected) < budget and len(selected) < limit:
        if groups[g]:
            selected.append(groups[g].pop(0))
            empty_streak = 0
        else:
            empty_streak += 1
        g = (g + 1) % len(groups)
        if empty_streak >= len(groups):
            break
    return selected


def select_area_random(unlabeled, budget, num_areas, seed=None):
    """Round-robin over ``num_areas`` groups (an image goes to group ``index % num_areas``), random within a group."""
    if len(unlabeled) < budget:
        return list(unlabeled)
    rng = _rng(seed)
    groups = [[] for _ in range(num_areas)]
    for idx in unlabeled:
        groups[idx % num_areas].append(idx)
    for group in groups:
        rng.shuffle(group)
    return _round_robin(groups, budget, len(unlabeled))


def select_area_uncertainty(unlabeled, budget, num_areas, uncertainties, seed=None):
    """Same grouping as ``select_area_random`` with each group ordered by descending uncertainty."""
    if len(unlabeled) < budget:
        return list(unlabeled)
    if uncertainties is None:
        return select_area_random(unlabeled, budget, num_areas, seed)
    groups = [[] for _ in range(num_areas)]
    for idx in unlabeled:
        groups[idx % num_areas].append(idx)
    groups = [sorted(group, key=lambda i: uncertainties[i], reverse=True) for group in groups]
    return _round_robin(groups, budget, len(unlabeled))


def select_diversity(unlabeled, budget, features, seed=None):
    """Greedy farthest-point sampling in feature space within the batch, starting from a random image."""
    if len(unlabeled) < budget:
        return list(unlabeled)
    rng = _rng(seed)
    if features is None:
        return rng.choice(unlabeled, size=budget, replace=False).tolist()

    feats = np.asarray(features)[unlabeled]
    first = rng.choice(len(unlabeled))
    selected = [unlabeled[first]]
    chosen = [feats[first]]
    remaining = [i for i in range(len(unlabeled)) if i != first]
    remaining_feats = feats[remaining]

    for _ in range(min(budget - 1, len(remaining))):
        min_dist = [min(np.linalg.norm(f - c) for c in chosen) for f in remaining_feats]
        best = int(np.argmax(min_dist))
        selected.append(unlabeled[remaining[best]])
        chosen.append(remaining_feats[best])
        remaining.pop(best)
        remaining_feats = np.delete(remaining_feats, best, axis=0)
    return selected
