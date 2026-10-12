"""USIMC: uncertainty-aware submodular mutual information on per-sample gradients (ALUNET, MIDL 2024).

Reference implementation: https://github.com/Berni1557/ALUNET. Requires ``submodlib``; without it the
strategy falls back to uncertainty sampling.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .simple import select_random, select_uncertainty

try:
    from submodlib.functions.facilityLocationVariantMutualInformation import \
        FacilityLocationVariantMutualInformationFunction
    SUBMODLIB_AVAILABLE = True
except ImportError:
    SUBMODLIB_AVAILABLE = False

MAX_GRADIENT_SAMPLES = 100
NUM_GRADIENT_PARAMS = 5000


def _per_sample_gradients(model, loader, device, num_params):
    """Flattened per-sample gradients of the BCE loss against the dataset masks, subsampled to ``num_params``."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    param_indices, gradients = None, []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        for i in range(images.shape[0]):
            output = model(images[i:i + 1])
            loss = F.binary_cross_entropy_with_logits(
                output.squeeze(1), targets[i:i + 1].squeeze(1).float(), reduction='sum')
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            flat = torch.cat([g.flatten() for g in grads if g is not None])
            if len(flat) > num_params:
                if param_indices is None:
                    param_indices = np.random.RandomState(42).choice(len(flat), size=num_params, replace=False)
                flat = flat[param_indices]
            gradients.append(flat.cpu().numpy())
    return np.array(gradients)


def select_usimc(unlabeled, budget, uncertainties, model, dataset, device, batch_size, seed=None):
    if len(unlabeled) < budget:
        return list(unlabeled)
    rng = np.random.RandomState(seed) if seed is not None else np.random

    def fallback(reason):
        print(f'USIMC: {reason}; falling back to uncertainty sampling.')
        if uncertainties is None:
            return select_random(unlabeled, budget, seed)
        return select_uncertainty(unlabeled, budget, uncertainties, seed)

    if not SUBMODLIB_AVAILABLE:
        return fallback('submodlib is not installed')

    if uncertainties is not None:
        unc = np.array([uncertainties[i] for i in unlabeled])
        candidates = np.argsort(unc)[-MAX_GRADIENT_SAMPLES:][::-1].tolist()
    else:
        unc = None
        candidates = rng.choice(len(unlabeled), size=min(MAX_GRADIENT_SAMPLES, len(unlabeled)),
                                replace=False).tolist()

    loader = DataLoader(Subset(dataset, [unlabeled[i] for i in candidates]), batch_size=batch_size, shuffle=False)
    try:
        gradients = _per_sample_gradients(model, loader, device, NUM_GRADIENT_PARAMS)
    except Exception as e:
        return fallback(f'gradient computation failed ({e})')

    if unc is not None and unc[candidates].sum() > 0:
        query_prob = unc[candidates] / unc[candidates].sum()
    else:
        query_prob = np.ones(len(candidates)) / len(candidates)
    num_queries = min(budget * 5, len(candidates))
    query = rng.choice(len(candidates), size=num_queries, replace=True, p=query_prob)

    try:
        objective = FacilityLocationVariantMutualInformationFunction(
            n=len(candidates), num_queries=num_queries, query_sijs=None, data=gradients,
            queryData=gradients[query], metric='cosine', queryDiversityEta=1.0)
        greedy = objective.maximize(
            budget=budget, optimizer='NaiveGreedy' if len(unlabeled) < 1000 else 'StochasticGreedy',
            stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, show_progress=False, epsilon=0.01)
    except Exception as e:
        return fallback(f'submodular maximization failed ({e})')
    return [unlabeled[candidates[i]] for i, _ in greedy]
