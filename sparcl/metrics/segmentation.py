"""Segmentation metrics: global Dice/IoU and the spatial reliability metrics (SI, PC) over grid bins."""

import torch


def calculate_metrics(pred, target, threshold=0.5, smooth=1e-6):
    """Batch-level Dice and IoU of the binarized prediction."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    dice = (2 * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)
    iou = intersection / (pred_binary.sum() + target.sum() - intersection + smooth)
    return dice.item(), iou.item()


def divide_image_into_areas(image_size=(512, 512), grid_width=7, grid_height=7):
    """Rectangular grid bins as ``(y0, y1, x0, x1, bin_index)``; the last row/column absorbs the remainder."""
    h, w = image_size
    area_h, area_w = h // grid_height, w // grid_width
    areas = []
    for i in range(grid_height):
        for j in range(grid_width):
            y1 = (i + 1) * area_h if i < grid_height - 1 else h
            x1 = (j + 1) * area_w if j < grid_width - 1 else w
            areas.append((i * area_h, y1, j * area_w, x1, i * grid_width + j))
    return areas


def accumulate_bin_stats(pred_masks, true_masks, areas, threshold=0.5):
    """Per-bin (TP, predicted, ground-truth) foreground pixel counts summed over a batch of [B, H, W] masks."""
    pred_bin = (pred_masks > threshold).float()
    true_bin = (true_masks > 0.5).float()
    tp = torch.zeros(len(areas), device=pred_masks.device)
    p = torch.zeros(len(areas), device=pred_masks.device)
    g = torch.zeros(len(areas), device=pred_masks.device)
    for y0, y1, x0, x1, idx in areas:
        pb, gb = pred_bin[:, y0:y1, x0:x1], true_bin[:, y0:y1, x0:x1]
        tp[idx] = (pb * gb).sum()
        p[idx] = pb.sum()
        g[idx] = gb.sum()
    return tp, p, g


def finalize_pooled_bin_metrics(tp, p, g, min_active=4, eps=1e-8):
    """Pooled per-bin Dice over the whole held-out set, summarized over bins with ground-truth foreground."""
    tp, p, g = tp.float(), p.float(), g.float()
    denom = p + g
    dice = torch.where(denom > 0, 2.0 * tp / denom, torch.zeros_like(tp))
    active = g > 0
    n_active = int(active.sum().item())
    result = {'D': dice, 'G': g, 'num_active_bins': n_active,
              'fp_pixels': float(p[(g == 0) & (p > 0)].sum().item())}
    if n_active == 0:
        result.update({'mean_bin_dice': 0.0, 'worst_bin_dice': 0.0, 'p10_bin_dice': 0.0,
                       'median_bin_dice': 0.0, 'max_bin_dice': 0.0, 'bin_std': 0.0,
                       'si_pooled': float('nan')})
        return result

    d = dice[active]
    mean = d.mean()
    std = d.std(unbiased=True) if d.numel() >= 2 else torch.zeros((), device=d.device)
    si = float('nan') if n_active < min_active or mean.item() <= eps else (std / (mean + eps)).item()
    result.update({
        'mean_bin_dice': mean.item(),
        'worst_bin_dice': d.min().item(),
        'p10_bin_dice': torch.quantile(d, 0.1).item(),
        'median_bin_dice': torch.median(d).item(),
        'max_bin_dice': d.max().item(),
        'bin_std': std.item(),
        'si_pooled': si,
    })
    return result


def per_image_spatial_inconsistency(pred_masks, true_masks, areas, threshold=0.5, min_active=2):
    """Spatial Inconsistency (Eq. SI of the paper) for each image of a [B, H, W] batch.

    For image x, D_m(x) is the Dice in bin m and A(x) = {m : |P_m| + |G_m| > 0} the active bins, which keeps
    false-positive-only and missed-lesion bins (both D_m = 0). SI(x) is the population coefficient of variation
    of D_m over A(x): NaN if |A(x)| < 2, 0 if the mean is 0.

    Returns per-image lists ``si``, ``mean_bin_dice`` (NaN if |A(x)| = 0) and ``num_active``.
    """
    pred_bin = (pred_masks > threshold).float()
    true_bin = (true_masks > 0.5).float()
    batch, n_bins = pred_bin.shape[0], len(areas)
    p = torch.zeros(batch, n_bins, device=pred_bin.device)
    g = torch.zeros(batch, n_bins, device=pred_bin.device)
    tp = torch.zeros(batch, n_bins, device=pred_bin.device)
    for y0, y1, x0, x1, idx in areas:
        pb, gb = pred_bin[:, y0:y1, x0:x1], true_bin[:, y0:y1, x0:x1]
        p[:, idx] = pb.sum(dim=(1, 2))
        g[:, idx] = gb.sum(dim=(1, 2))
        tp[:, idx] = (pb * gb).sum(dim=(1, 2))

    denom = p + g
    active = denom > 0
    dice = torch.where(active, 2.0 * tp / torch.clamp(denom, min=1.0), torch.zeros_like(tp))

    si, mean_bin_dice, num_active = [], [], []
    for b in range(batch):
        d = dice[b][active[b]]
        k = d.numel()
        num_active.append(k)
        if k == 0:
            si.append(float('nan'))
            mean_bin_dice.append(float('nan'))
            continue
        mean = d.mean().item()
        mean_bin_dice.append(mean)
        if k < min_active:
            si.append(float('nan'))
        elif mean == 0.0:
            si.append(0.0)
        else:
            si.append((d.std(unbiased=False) / d.mean()).item())
    return {'si': si, 'mean_bin_dice': mean_bin_dice, 'num_active': num_active}


def performance_coverage(dice_pre, dice_post, active):
    """Performance Coverage: mean over ground-truth active bins of sqrt(max(D_post - D_pre, 0))."""
    if int(active.sum().item()) == 0:
        return float('nan')
    gain = torch.sqrt(torch.clamp(dice_post - dice_pre, min=0.0))
    return gain[active].float().mean().item()
