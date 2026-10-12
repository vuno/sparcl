"""Detection metrics: COCO-style mAP@[.5:.95], AP50, AP75 and FROC."""

import numpy as np
import torch


def _to_numpy(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def box_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def _iou_matrix(pred_boxes, target_boxes):
    return np.array([[box_iou(p, t) for t in target_boxes] for p in pred_boxes]).reshape(
        len(pred_boxes), len(target_boxes))


def _greedy_match(pred_scores, iou_matrix, iou_threshold):
    """Match predictions (in descending score order) to unmatched targets; returns (order, tp flags, ious)."""
    matched, tp_flags, ious = set(), [], []
    order = np.argsort(pred_scores)[::-1]
    for i in order:
        best_iou, best_j = 0.0, -1
        for j in range(iou_matrix.shape[1]):
            if j in matched:
                continue
            iou = iou_matrix[i, j]
            if iou > best_iou and iou >= iou_threshold:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched.add(best_j)
            tp_flags.append(1)
            ious.append(best_iou)
        else:
            tp_flags.append(0)
    return order, tp_flags, ious


def _collect_detections(predictions, targets, iou_threshold):
    all_scores, all_tp, total_gt = [], [], 0
    for pred, target in zip(predictions, targets):
        pred_boxes = _to_numpy(pred['boxes']).reshape(-1, 4)
        pred_scores = _to_numpy(pred['scores']).reshape(-1)
        target_boxes = _to_numpy(target['boxes']).reshape(-1, 4)
        total_gt += len(target_boxes)
        if len(pred_boxes) == 0:
            continue
        if len(target_boxes) == 0:
            all_scores.extend(pred_scores.tolist())
            all_tp.extend([0] * len(pred_scores))
            continue
        order, tp_flags, _ = _greedy_match(pred_scores, _iou_matrix(pred_boxes, target_boxes), iou_threshold)
        all_scores.extend(pred_scores[order].tolist())
        all_tp.extend(tp_flags)
    return all_scores, all_tp, total_gt


def _ap_from_pr(precisions, recalls):
    """101-point interpolated AP."""
    if len(precisions) == 0:
        return 0.0
    order = np.argsort(recalls)
    precisions = np.concatenate([[0], np.asarray(precisions)[order], [0]])
    recalls = np.concatenate([[0], np.asarray(recalls)[order], [1]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    return float(np.mean([precisions[recalls >= t].max() if (recalls >= t).any() else 0.0
                          for t in np.linspace(0, 1, 101)]))


def average_precision(predictions, targets, iou_threshold):
    scores, tp, total_gt = _collect_detections(predictions, targets, iou_threshold)
    if total_gt == 0:
        return 1.0 if len(scores) == 0 else 0.0
    if len(scores) == 0:
        return 0.0
    tp = np.array(tp)[np.argsort(scores)[::-1]]
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
    return _ap_from_pr(cum_tp / (cum_tp + cum_fp + 1e-8), cum_tp / total_gt)


def froc(predictions, targets, num_images, iou_threshold=0.5, fp_rates=(0.125, 0.25, 0.5, 1, 2, 4, 8)):
    """Mean sensitivity at the given false positives per image, and the normalized area under the FROC curve."""
    scores, tp, total_gt = _collect_detections(predictions, targets, iou_threshold)
    if total_gt == 0:
        value = 1.0 if len(scores) == 0 else 0.0
        return {'froc_score': value, 'auc_froc': value}
    if len(scores) == 0:
        return {'froc_score': 0.0, 'auc_froc': 0.0}

    tp = np.array(tp)[np.argsort(scores)[::-1]]
    sensitivity = np.cumsum(tp) / total_gt
    fp_per_image = np.cumsum(1 - tp) / num_images
    froc_score = np.mean([sensitivity[fp_per_image <= r].max() if (fp_per_image <= r).any() else 0.0
                          for r in fp_rates])

    max_rate = max(fp_rates)
    valid = fp_per_image <= max_rate
    auc = 0.0
    if valid.any():
        auc = np.trapz(np.concatenate([[0], sensitivity[valid]]),
                       np.concatenate([[0], fp_per_image[valid]])) / max_rate
    return {'froc_score': float(froc_score), 'auc_froc': float(auc)}


def calculate_detection_metrics(predictions, targets, iou_threshold=0.5, score_threshold=0.1):
    """mAP, AP50, AP75, FROC and per-image precision/recall/IoU (predictions with score >= score_threshold)."""
    precisions, recalls, ious = [], [], []
    for pred, target in zip(predictions, targets):
        pred_boxes = _to_numpy(pred['boxes']).reshape(-1, 4)
        pred_scores = _to_numpy(pred['scores']).reshape(-1)
        target_boxes = _to_numpy(target['boxes']).reshape(-1, 4)
        keep = pred_scores >= score_threshold
        pred_boxes, pred_scores = pred_boxes[keep], pred_scores[keep]

        if len(pred_boxes) == 0:
            precision, recall, avg_iou = (0.0, 0.0, 0.0) if len(target_boxes) > 0 else (1.0, 1.0, 1.0)
        elif len(target_boxes) == 0:
            precision, recall, avg_iou = 0.0, 1.0, 0.0
        else:
            _, tp_flags, matched_ious = _greedy_match(
                pred_scores, _iou_matrix(pred_boxes, target_boxes), iou_threshold)
            n_tp = sum(tp_flags)
            precision = n_tp / (len(pred_boxes) + 1e-8)
            recall = n_tp / (len(target_boxes) + 1e-8)
            avg_iou = np.mean(matched_ious) if matched_ious else 0.0
        precisions.append(precision)
        recalls.append(recall)
        ious.append(avg_iou)

    froc_metrics = froc(predictions, targets, len(predictions), iou_threshold=0.5)
    return {
        'precision': float(np.mean(precisions)) if precisions else 0.0,
        'recall': float(np.mean(recalls)) if recalls else 0.0,
        'avg_iou': float(np.mean(ious)) if ious else 0.0,
        'mAP': float(np.mean([average_precision(predictions, targets, t) for t in np.arange(0.5, 1.0, 0.05)])),
        'AP50': average_precision(predictions, targets, 0.5),
        'AP75': average_precision(predictions, targets, 0.75),
        'froc_score': froc_metrics['froc_score'],
        'auc_froc': froc_metrics['auc_froc'],
    }
