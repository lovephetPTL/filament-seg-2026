"""Paired bootstrap over validation photos (numpy only).

counts_* : arrays of shape (n_photos, 4) with columns [TP, sum of TP IoUs, FP, FN], pooled over all annotator-images
of a photo and aligned on the same photos. A resample draws photos with replacement and recomputes the pooled PQ of
BOTH models on the SAME photos, so the interval of the difference is paired. It covers the sampling of validation photos
only; it does not cover the variance between training runs (each model was trained once).
"""
import numpy as np


def pq_from_sums(s):
    tp, iou, fp, fn = (float(x) for x in s)
    denom = tp + 0.5 * fp + 0.5 * fn
    return iou / denom if denom > 0 else 0.0


def paired_bootstrap(counts_a, counts_b, n_boot=10000, seed=0):
    a, b = np.asarray(counts_a, float), np.asarray(counts_b, float)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 4, "need aligned (n_photos, 4) arrays"
    n = len(a)
    rng = np.random.default_rng(seed)
    pa, pb = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        pa[i] = pq_from_sums(a[idx].sum(0))
        pb[i] = pq_from_sums(b[idx].sum(0))
    d = pb - pa
    q = lambda x: [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]
    return {
        "n_photos": n, "n_boot": n_boot, "seed": seed,
        "PQ_A": round(pq_from_sums(a.sum(0)), 4), "PQ_B": round(pq_from_sums(b.sum(0)), 4),
        "diff_B_minus_A": round(pq_from_sums(b.sum(0)) - pq_from_sums(a.sum(0)), 4),
        "ci95_A": q(pa), "ci95_B": q(pb), "ci95_diff": q(d),
        "share_of_resamples_B_better": round(float((d > 0).mean()), 3),
    }
