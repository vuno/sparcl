"""Per-round training with early stopping and held-out evaluation."""

import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import detection_collate_fn
from .losses import combo_loss
from .metrics import (accumulate_bin_stats, calculate_detection_metrics, calculate_metrics,
                      finalize_pooled_bin_metrics, per_image_spatial_inconsistency, performance_coverage)

DETECTION_KEYS = ('mAP', 'AP50', 'AP75', 'froc_score', 'auc_froc', 'precision', 'avg_iou')


def _round_logger(output_dir, round_num):
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    logger = logging.getLogger(f'round{round_num}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    handler = logging.FileHandler(os.path.join(output_dir, 'logs', f'round{round_num}.log'))
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


def _train_epoch(model, loader, optimizer, scheduler, device, task):
    model.train()
    total = 0.0
    for images, targets in loader:
        if task == 'detection':
            images = [img.to(device) for img in images]
            targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]
            loss = model(images, targets=targets)['loss']
        else:
            images, targets = images.to(device), targets.to(device)
            loss = combo_loss(model(images), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _evaluate_detection(model, loader, device):
    """Loss and detection metrics averaged over held-out batches."""
    sums = dict.fromkeys(DETECTION_KEYS, 0.0)
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]
        model.eval()
        predictions = model(images)
        model.train()
        total_loss += model(images, targets=targets)['loss'].item()
        model.eval()
        metrics = calculate_detection_metrics(predictions, targets, iou_threshold=0.5)
        for key in DETECTION_KEYS:
            sums[key] += metrics[key]
    n = max(len(loader), 1)
    result = {key: value / n for key, value in sums.items()}
    result['loss'] = total_loss / len(loader)
    return result, None


@torch.no_grad()
def _evaluate_segmentation(model, loader, device, areas):
    """Loss, Dice/IoU (batch averages), pooled per-bin metrics and per-image Spatial Inconsistency."""
    model.eval()
    total_loss = dice_sum = iou_sum = 0.0
    tp = p = g = None
    si, mean_bin_dice = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        probs = model(images)
        total_loss += combo_loss(probs, masks).item()
        dice, iou = calculate_metrics(probs, masks)
        dice_sum += dice
        iou_sum += iou

        b_tp, b_p, b_g = accumulate_bin_stats(probs.squeeze(1), masks.squeeze(1), areas)
        tp, p, g = (b_tp, b_p, b_g) if tp is None else (tp + b_tp, p + b_p, g + b_g)
        per_image = per_image_spatial_inconsistency(probs.squeeze(1), masks.squeeze(1), areas)
        si.extend(per_image['si'])
        mean_bin_dice.extend(per_image['mean_bin_dice'])

    n = len(loader)
    pooled = finalize_pooled_bin_metrics(tp, p, g)
    si = np.asarray(si, dtype=np.float64)
    defined = ~np.isnan(si)
    result = {
        'loss': total_loss / n,
        'dice': dice_sum / n,
        'iou': iou_sum / n,
        'si': float(si[defined].mean()) if defined.any() else float('nan'),
        'si_undefined_fraction': float(1.0 - defined.mean()),
        'mean_bin_dice_per_image': float(np.nanmean(mean_bin_dice)) if mean_bin_dice else float('nan'),
    }
    for key in ('mean_bin_dice', 'worst_bin_dice', 'p10_bin_dice', 'median_bin_dice', 'max_bin_dice',
                'bin_std', 'si_pooled', 'num_active_bins', 'fp_pixels'):
        result[key] = pooled[key]
    bin_state = {'D': pooled['D'].cpu(), 'G': pooled['G'].cpu()}
    return result, bin_state


def train_round(model, train_dataset, val_dataset, device, task, areas, args, round_num, output_dir,
                prev_bin_state=None):
    """Train on the labeled set with early stopping on the held-out set.

    Early stopping monitors the held-out loss (segmentation) or AP50 (detection). The returned metrics are those
    of the best epoch. The returned model keeps the weights of the last epoch unless ``args.restore_best_weights``.
    For detection an epoch only counts as an improvement once AP50 exceeds 0; if that never happens, the
    last-epoch metrics are returned and no checkpoint is written.

    Returns:
        (metrics dict, bin_state of the best epoch, checkpoint path or None)
    """
    if task == 'detection':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, collate_fn=detection_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                pin_memory=True, collate_fn=detection_collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                pin_memory=True, persistent_workers=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=args.train_epochs,
                                                    steps_per_epoch=len(train_loader))
    logger = _round_logger(output_dir, round_num)
    logger.info(f'round {round_num}: train={len(train_dataset)} held-out={len(val_dataset)}')

    best_score = 0.0 if task == 'detection' else -float('inf')
    best_epoch, patience = 0, 0
    best_metrics, best_bin_state, best_weights = None, None, None
    progress = tqdm(range(args.train_epochs), desc=f'round {round_num}', leave=False)
    for epoch in progress:
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, device, task)
        if task == 'detection':
            metrics, bin_state = _evaluate_detection(model, val_loader, device)
            score = metrics['AP50']
        else:
            metrics, bin_state = _evaluate_segmentation(model, val_loader, device, areas)
            prev_dice = prev_bin_state['D'] if prev_bin_state is not None else torch.zeros_like(bin_state['D'])
            metrics['pc'] = performance_coverage(prev_dice, bin_state['D'], bin_state['G'] > 0)
            score = -metrics['loss']
        metrics['train_loss'] = train_loss

        if score > best_score + args.min_delta:
            best_score, best_epoch, patience = score, epoch + 1, 0
            best_metrics, best_bin_state = dict(metrics), bin_state
            if args.restore_best_weights:
                best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        logger.info(f'epoch {epoch + 1}: ' + ', '.join(
            f'{k}={v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float))))
        progress.set_postfix(train_loss=f'{train_loss:.4f}', score=f'{score:.4f}', best=f'{best_score:.4f}')
        if patience >= args.patience:
            logger.info(f'early stop at epoch {epoch + 1}; best epoch {best_epoch}')
            break
    progress.close()

    if best_weights is not None:
        model.load_state_dict(best_weights)
    if best_metrics is None:
        best_metrics, best_bin_state = metrics, bin_state
        best_metrics['best_epoch'] = 0
        return best_metrics, best_bin_state, None
    best_metrics['best_epoch'] = best_epoch

    checkpoint_dir = os.path.join(output_dir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'round{round_num}.pth')
    torch.save({'model_state_dict': model.state_dict(), 'best_epoch': best_epoch, 'round': round_num},
               checkpoint_path)
    return best_metrics, best_bin_state, checkpoint_path
