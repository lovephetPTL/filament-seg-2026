"""Our own Panoptic Quality (PQ) implementation.

It reproduces the organisers Self-Evaluation notebook exactly on our validation
split (both give 0.3825 for the baseline), see README. For official numbers use
the organisers notebook; this file exists for fast CPU-only experiments.
"""
import numpy as np


def match_instances(pred_labeled, gt_lists, iou_thresh=0.5):
    """Greedy one-to-one matching (highest IoU first), IoU must exceed iou_thresh.

    pred_labeled : (H, W) int array, 0 = background, 1..n = predicted pieces
    gt_lists     : list of 1-D index arrays (flat pixel positions of each GT filament)
    Returns (TP, sum of TP IoUs, n_pred, n_gt)
    """
    n_pred = int(pred_labeled.max())
    n_gt = len(gt_lists)
    if n_pred == 0 or n_gt == 0:
        return 0, 0.0, n_pred, n_gt

    pred_flat = pred_labeled.ravel()
    pred_area = np.bincount(pred_flat, minlength=n_pred + 1)[1:]

    candidates = []
    for g_i, idx in enumerate(gt_lists):
        inter = np.bincount(pred_flat[idx], minlength=n_pred + 1)[1:]
        union = pred_area + idx.size - inter
        iou = inter / np.maximum(union, 1)
        for p_i in np.flatnonzero(iou > iou_thresh):
            candidates.append((float(iou[p_i]), int(p_i), g_i))

    candidates.sort(reverse=True)
    used_p, used_g = set(), set()
    tp, sum_iou = 0, 0.0
    for iou, p_i, g_i in candidates:
        if p_i in used_p or g_i in used_g:
            continue
        used_p.add(p_i)
        used_g.add(g_i)
        tp += 1
        sum_iou += iou
    return tp, sum_iou, n_pred, n_gt


def pq_from_counts(tp, sum_iou, n_pred, n_gt):
    """PQ = sum(IoU of TP) / (TP + 0.5 FP + 0.5 FN). Returns None if undefined."""
    fp = n_pred - tp
    fn = n_gt - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return None if denom == 0 else sum_iou / denom
