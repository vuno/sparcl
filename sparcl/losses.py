import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.clamp(pred.reshape(-1), 0, 1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    pred = torch.clamp(pred.reshape(-1), 0, 1)
    target = target.reshape(-1)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def combo_loss(pred, target, dice_weight=0.4, bce_weight=0.3, focal_weight=0.3):
    """Segmentation loss: 0.4 * Dice + 0.3 * BCE + 0.3 * Focal(alpha=0.75, gamma=3)."""
    return (dice_weight * dice_loss(pred, target)
            + bce_weight * F.binary_cross_entropy(pred, target)
            + focal_weight * focal_loss(pred, target, alpha=0.75, gamma=3.0))
