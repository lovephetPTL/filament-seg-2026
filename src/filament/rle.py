"""COCO run-length encoding of masks and writing of the submission CSV."""
import csv

import numpy as np
from pycocotools import mask as mask_util

from .config import IMG_SIZE


def encode_mask(mask):
    """Boolean/0-1 (H, W) mask -> COCO compressed RLE `counts` string."""
    m = np.asfortranarray(np.asarray(mask).astype(np.uint8))
    return mask_util.encode(m)["counts"].decode("utf-8")


def decode_rle(counts, size=IMG_SIZE):
    """COCO RLE `counts` string -> boolean (size, size) mask."""
    rle = {"size": [size, size], "counts": counts.encode("utf-8")}
    return mask_util.decode(rle).astype(bool)


def write_submission(rows, path):
    """rows: iterable of (filament_id, rle_counts). Writes the Kaggle CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
