"""Core-Set k-center greedy selection (Sener & Savarese, ICLR 2018)."""

import numpy as np


def select_coreset(unlabeled, labeled, budget, features, seed=None):
    """Greedily add the image farthest (L2 on unit-normalized features) from the labeled set and the current batch."""
    if len(unlabeled) < budget:
        return list(unlabeled)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    if features is None:
        return rng.choice(unlabeled, size=budget, replace=False).tolist()

    def normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 0, norms, 1.0)

    features = np.asarray(features)
    pool_feats = normalize(features[unlabeled])
    labeled_feats = normalize(features[labeled]) if len(labeled) > 0 else np.zeros((0, features.shape[1]))

    if len(labeled_feats) > 0:
        dist_to_labeled = np.array([np.min(np.linalg.norm(f - labeled_feats, axis=1)) for f in pool_feats])
        first = int(np.argmax(dist_to_labeled))
    else:
        first = rng.choice(len(unlabeled))

    selected = [unlabeled[first]]
    batch_feats = [pool_feats[first]]
    remaining = [i for i in range(len(unlabeled)) if i != first]

    for _ in range(min(budget - 1, len(remaining))):
        min_dist = []
        for i in remaining:
            d_old = np.min(np.linalg.norm(pool_feats[i] - labeled_feats, axis=1)) if len(labeled_feats) else np.inf
            d_new = np.min(np.linalg.norm(pool_feats[i] - np.asarray(batch_feats), axis=1))
            min_dist.append(min(d_old, d_new))
        best = remaining[int(np.argmax(min_dist))]
        selected.append(unlabeled[best])
        batch_feats.append(pool_feats[best])
        remaining.remove(best)
    return selected
