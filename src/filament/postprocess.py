"""Binary mask -> list of filament instances.

Baseline (public score 0.31): connected components, drop pieces < MIN_SIZE px.
Final   (public score 0.32): same, but nearby pieces are first GROUPED with a
morphological closing (k = 31). The closing is used only to decide which
pixels belong together; the submitted pixels are the original ones.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import CLOSING_KERNEL, MIN_SIZE


def extract_instances(binary, min_size=MIN_SIZE, closing_kernel=None):
    """Return a list of boolean masks, one per predicted filament.

    Parameters
    ----------
    binary : (H, W) array with values in {0, 1}
    min_size : pieces with fewer *original* pixels are dropped
    closing_kernel : None -> plain connected components (baseline);
                     int k -> group pieces with an elliptical closing of size k
                     (pixels are never added: labels are masked by `binary`)
    """
    b = (np.asarray(binary) > 0).astype(np.uint8)
    if closing_kernel:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        lab, n = ndi.label(cv2.morphologyEx(b, cv2.MORPH_CLOSE, kern))
        lab = lab * b          # closing is extensive, so every original pixel keeps a label
    else:
        lab, n = ndi.label(b)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    return [lab == i for i in range(1, n + 1) if sizes[i] >= min_size]


def extract_instances_final(binary):
    """The configuration of the final submission (closing k=31, min size 120)."""
    return extract_instances(binary, MIN_SIZE, CLOSING_KERNEL)
