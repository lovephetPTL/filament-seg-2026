"""Validation PQ in the same way as the organisers Self-Evaluation notebook (numpy only).

Every validation ANNOTATOR-IMAGE is scored on its own against the predictions of its photo;
TP IoUs, FP and FN are pooled over all annotator-images and divided once.
"""
import numpy as np

from .metrics import match_instances, pq_from_counts


def gt_pixel_lists(coco, img_id):
    out = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
        idx = np.flatnonzero(coco.annToMask(ann))
        if idx.size:
            out.append(idx)
    return out


def label_image(instances, size=2048):
    """List of disjoint boolean masks -> int32 label image (0 = background)."""
    lab = np.zeros((size, size), np.int32)
    for i, m in enumerate(instances, start=1):
        lab[m] = i
    return lab


def evaluate_pq(coco, val_ids, instances_by_photo, photo_of):
    """val_ids: annotator-image ids; instances_by_photo: {photo: [bool masks]}.
    Returns a dict with PQ, TP, FP, FN, SQ, RQ and the number of predicted pieces."""
    tp = fp = fn = 0
    sum_iou = 0.0
    labels = {}
    for img_id in val_ids:
        gts = gt_pixel_lists(coco, img_id)
        if not gts:                                   # same as the organisers: no GT rows -> not scored
            continue
        photo = photo_of(img_id)
        if photo not in labels:
            labels[photo] = label_image(instances_by_photo[photo])
        t, s, n_pred, n_gt = match_instances(labels[photo], gts)
        tp += t
        sum_iou += s
        fp += n_pred - t
        fn += n_gt - t
    pq = pq_from_counts(tp, sum_iou, tp + fp, tp + fn)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    return {"PQ": round(pq, 4), "TP": tp, "FP": fp, "FN": fn,
            "SQ": round(sum_iou / max(tp, 1), 3), "RQ": round(rq, 3),
            "pred_pieces": sum(len(v) for v in instances_by_photo.values())}


def per_photo_counts(coco, val_ids, instances_by_photo, photo_of):
    """{photo: np.array([TP, sum of TP IoUs, FP, FN])}, pooled over the annotator-images of the photo.
    The sum over all photos reproduces `evaluate_pq`."""
    out, labels = {}, {}
    for img_id in val_ids:
        gts = gt_pixel_lists(coco, img_id)
        if not gts:
            continue
        photo = photo_of(img_id)
        if photo not in labels:
            labels[photo] = label_image(instances_by_photo[photo])
        t, s_iou, n_pred, n_gt = match_instances(labels[photo], gts)
        out.setdefault(photo, np.zeros(4))
        out[photo] += (t, s_iou, n_pred - t, n_gt - t)
    return out
