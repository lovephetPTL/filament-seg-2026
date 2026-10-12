"""Example figure: ground truth of all annotators (cyan) and predicted pieces (red) on H-alpha crops.
Numpy / OpenCV only; matplotlib is imported by the script that saves the figure."""
import cv2
import numpy as np


def crop_box(mask, size=768, img_size=2048):
    """Deterministic crop window (y0, x0) of `size` px centred on the centre of mass of `mask`
    (or on the image centre if the mask is empty), clipped to the image."""
    ys, xs = np.nonzero(mask)
    cy, cx = (int(ys.mean()), int(xs.mean())) if ys.size else (img_size // 2, img_size // 2)
    y0 = int(np.clip(cy - size // 2, 0, img_size - size))
    x0 = int(np.clip(cx - size // 2, 0, img_size - size))
    return y0, x0


def stretch(gray, lo=1, hi=99):
    """Contrast stretch for display only (percentiles of the crop)."""
    a, b = np.percentile(gray, [lo, hi])
    return np.clip((gray.astype(np.float32) - a) / max(b - a, 1e-6), 0, 1)


def overlay(gray_crop, gt_masks, pred_masks, thickness=1):
    """RGB float image: stretched grey crop with annotator contours in cyan and predicted contours in red."""
    base = (stretch(gray_crop) * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1).copy()
    for masks, colour in ((gt_masks, (0, 255, 255)), (pred_masks, (255, 0, 0))):
        for m in masks:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cnts, -1, colour, thickness)
    return rgb
