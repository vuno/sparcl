#!/usr/bin/env python3
"""Active learning loop for SPARCL and the baseline strategies.

Example:
    python run_al.py --dataset siim --siim_root /path/to/siim --mode sparcl --grid 7 \
        --num_samples 15 --round_num 25 --seed 0 --output_dir results
"""

import argparse
import json
import os
from datetime import datetime

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from sparcl.acquisition import (PAAL, SEGMENTATION_ONLY, STRATEGIES, box_lesion_distribution,
                                mask_lesion_distribution, score_detection, score_segmentation,
                                select_area_random, select_area_uncertainty, select_coreset, select_diversity,
                                select_loss_prediction, select_random, select_sparcl, select_taudis,
                                select_uncertainty, select_usimc)
from sparcl.data import augmentation_disabled, create_al_datasets, detection_collate_fn, task_type
from sparcl.metrics import divide_image_into_areas
from sparcl.models import create_model
from sparcl.training import train_round
from sparcl.uncertainty import detection_uncertainty, segmentation_uncertainty

NEEDS_PREDICTIONS = {'taudis', 'paal'}
NEEDS_FEATURES = {'diversity', 'coreset', 'taudis'}
NEEDS_UNCERTAINTY = {'uncertainty', 'uncertainty_area', 'diversity', 'coreset', 'taudis', 'usimc', 'paal'}


def parse_args():
    parser = argparse.ArgumentParser(description='SPARCL active learning')
    parser.add_argument('--dataset', required=True, choices=['siim', 'ddr', 'vindr', 'chestxdet10'])
    parser.add_argument('--mode', required=True, choices=STRATEGIES)
    parser.add_argument('--full_training', action='store_true',
                        help='Train once on a random 80%% of the pool and evaluate on the other 20%%.')

    parser.add_argument('--siim_root')
    parser.add_argument('--siim_positive_only', action='store_true')
    parser.add_argument('--ddr_root')
    parser.add_argument('--ddr_lesion', default='MA', choices=['EX', 'HE', 'MA', 'SE'])
    parser.add_argument('--ddr_positive_only', action='store_true')
    parser.add_argument('--vindr_root')
    parser.add_argument('--chestxdet10_root')
    parser.add_argument('--target_lesion', default='dispersed',
                        help="Detection target: a lesion name, 'dispersed', 'concentrated' or 'all'.")
    parser.add_argument('--min_radiologist_agreement', type=int, default=1)
    parser.add_argument('--include_negative', action='store_true')
    parser.add_argument('--num_validation_samples', type=int, default=100,
                        help='Held-out size for SIIM-ACR and DDR (VinDr-CXR / ChestX-Det10 use the test split).')

    parser.add_argument('--round_num', type=int, default=25)
    parser.add_argument('--num_samples', type=int, default=15, help='Images acquired per round (B).')
    parser.add_argument('--grid', type=int, default=7, help='K = L of the K x L spatial grid.')
    parser.add_argument('--lambda_max', type=float, default=1.0)
    parser.add_argument('--candidate_ratio', type=float, default=5.0, help='r: SPARCL candidate set size is r * B.')

    parser.add_argument('--encoder', default='efficientnet-b4', choices=['efficientnet-b4', 'resnet50'])
    parser.add_argument('--train_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--restore_best_weights', action='store_true',
                        help='Restore the best-epoch weights after training (default: keep the last epoch).')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_seed', type=int, default=42, help='Seed of the first-round selection.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    args.task = task_type(args.dataset)
    if args.task == 'detection' and args.mode in SEGMENTATION_ONLY:
        parser.error(f'--mode {args.mode} is only defined for segmentation')
    if args.candidate_ratio < 1:
        parser.error('--candidate_ratio must be >= 1')
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pool_statistics(args, model, pool, device):
    """Uncertainty (and, when required, predictions / features) for every pool image, indexed by dataset index."""
    need_predictions = args.mode in NEEDS_PREDICTIONS
    need_features = args.mode in NEEDS_FEATURES
    if args.task == 'detection':
        loader = DataLoader(pool, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=detection_collate_fn)
        stats = detection_uncertainty(model, loader, device, need_predictions, need_features)
    else:
        loader = DataLoader(pool, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        stats = segmentation_uncertainty(model, loader, device, need_predictions, need_features)
    if args.task == 'segmentation' and 'predictions' in stats:
        stats['probs'] = stats['predictions']
    return stats


def lesion_distributions(args, pool, labeled, areas, cache):
    """Ground-truth lesion distributions g(x) of the lesion-bearing labeled images (cached per index)."""
    dists = []
    for idx in labeled:
        if idx not in cache:
            if args.task == 'detection':
                cache[idx] = box_lesion_distribution(pool.gt_boxes(idx), areas)
            else:
                cache[idx] = mask_lesion_distribution(pool[idx][1].squeeze(0).numpy(), areas)
        if cache[idx] is not None:
            dists.append(cache[idx])
    return np.array(dists).reshape(-1, len(areas))


def acquire(args, round_idx, model, pool, labeled, areas, device, state):
    """Select ``args.num_samples`` new dataset indices; returns (indices, diagnostics)."""
    labeled_set = set(labeled)
    unlabeled = [i for i in range(len(pool)) if i not in labeled_set]
    budget, mode = args.num_samples, args.mode
    seed = args.init_seed if round_idx == 0 else None
    first_round = len(labeled) == 0

    if mode == 'sparcl':
        if first_round or len(unlabeled) <= budget:
            return select_random(unlabeled, budget, seed), {}
        if args.task == 'detection':
            with augmentation_disabled(pool):
                U, q = score_detection(model, pool, unlabeled, areas, device, args.batch_size, detection_collate_fn)
        else:
            U, q = score_segmentation(model, pool, unlabeled, areas, device, args.batch_size)
        g = lesion_distributions(args, pool, labeled, areas, state['lesion_cache'])
        return select_sparcl(unlabeled, budget, U, q, g, args.lambda_max, args.candidate_ratio)

    if mode == 'random':
        return select_random(unlabeled, budget, seed), {}
    if mode == 'area_random':
        return select_area_random(unlabeled, budget, len(areas), seed), {}
    if mode == 'loss_prediction':
        return select_loss_prediction(unlabeled, budget, model, pool, device, args.batch_size, seed), {}

    stats = {}
    if not first_round and mode in NEEDS_UNCERTAINTY:
        stats = pool_statistics(args, model, pool, device)
    unc = stats.get('uncertainties')

    if mode == 'uncertainty':
        return select_uncertainty(unlabeled, budget, unc, seed), {}
    if mode == 'uncertainty_area':
        return select_area_uncertainty(unlabeled, budget, len(areas), unc, seed), {}
    if mode == 'diversity':
        return select_diversity(unlabeled, budget, stats.get('features'), seed), {}
    if mode == 'coreset':
        return select_coreset(unlabeled, labeled, budget, stats.get('features'), seed), {}
    if mode == 'taudis':
        return select_taudis(unlabeled, budget, unc, stats.get('features'), stats.get('probs'), seed=seed), {}
    if mode == 'usimc':
        if args.task == 'detection':
            print('USIMC gradients are defined for the segmentation loss; using uncertainty sampling.')
            return select_uncertainty(unlabeled, budget, unc, seed), {}
        return select_usimc(unlabeled, budget, unc, model, pool, device, args.batch_size, seed), {}
    if mode == 'paal':
        return state['paal'].select(unlabeled, budget, pool, stats.get('probs'), unc, seed), {}
    raise ValueError(f'Unknown mode: {mode}')


def make_output_dir(args):
    name = f"{args.mode}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = os.path.join(args.output_dir, args.dataset, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return out


def save_results(results, output_dir):
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)


def grid_areas(args, pool):
    size = pool.target_size if args.task == 'detection' else (512, 512)
    return divide_image_into_areas(image_size=size, grid_width=args.grid, grid_height=args.grid)


def run_full_training(args, device, output_dir):
    pool, _ = create_al_datasets(args)
    n_train = int(0.8 * len(pool))
    train, val = random_split(pool, [n_train, len(pool) - n_train],
                              generator=torch.Generator().manual_seed(args.seed))
    model = create_model(args, device)
    metrics, _, _ = train_round(model, train, val, device, args.task, grid_areas(args, pool), args, 1, output_dir)
    save_results([{'mode': 'full_training', 'num_train': n_train, 'num_eval': len(val), 'metrics': metrics}],
                 output_dir)
    print(json.dumps(metrics, indent=2, default=float))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    output_dir = make_output_dir(args)
    if args.full_training:
        run_full_training(args, device, output_dir)
        return

    pool, heldout = create_al_datasets(args)
    areas = grid_areas(args, pool)
    print(f'pool={len(pool)} held-out={len(heldout)} task={args.task} mode={args.mode} output={output_dir}')

    labeled, results = [], []
    prev_bin_state, prev_checkpoint = None, None
    state = {'lesion_cache': {}, 'paal': PAAL(device) if args.mode == 'paal' else None}

    for round_idx in range(args.round_num):
        scoring_model = create_model(args, device, prev_checkpoint)
        new, diagnostics = acquire(args, round_idx, scoring_model, pool, labeled, areas, device, state)
        del scoring_model
        labeled.extend(new)

        model = create_model(args, device)
        metrics, bin_state, checkpoint = train_round(
            model, Subset(pool, labeled), heldout, device, args.task, areas, args, round_idx + 1, output_dir,
            prev_bin_state)
        if state['paal'] is not None:
            state['paal'].fit(model, Subset(pool, labeled))

        results.append({'round': round_idx + 1, 'num_labeled': len(labeled), 'selected': [int(i) for i in new],
                        'metrics': metrics, 'acquisition': diagnostics})
        save_results(results, output_dir)
        summary = (f"mAP={metrics['mAP']:.4f} AP50={metrics['AP50']:.4f} FROC={metrics['froc_score']:.4f}"
                   if args.task == 'detection' else
                   f"Dice={metrics['dice']:.4f} SI={metrics['si']:.4f} PC={metrics['pc']:.4f}")
        lam = f" lambda={diagnostics['lambda']:.3f}" if 'lambda' in diagnostics else ''
        print(f'round {round_idx + 1}/{args.round_num}: labeled={len(labeled)} {summary}{lam}')

        prev_bin_state, prev_checkpoint = bin_state, checkpoint


if __name__ == '__main__':
    main()
