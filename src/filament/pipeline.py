"""End-to-end: folder of test images -> submission CSV (+ self checks)."""
import os

import numpy as np
from tqdm import tqdm

from .config import CLOSING_KERNEL, IMG_SIZE, MIN_SIZE, THRESHOLD
from .infer import predict_binary_mask
from .postprocess import extract_instances
from .rle import decode_rle, encode_mask, write_submission

IMAGE_EXT = (".jpeg", ".jpg", ".png")


def make_submission(image_dir, model, device, out_csv,
                    closing_kernel=CLOSING_KERNEL, min_size=MIN_SIZE, threshold=THRESHOLD):
    """closing_kernel=None reproduces the baseline file (public 0.31);
    closing_kernel=31 reproduces the final file (public 0.32).
    Returns a dict with the self-check results."""
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXT))
    rows, per_image, overlap_bad, decode_bad = [], {}, 0, 0
    for fn in tqdm(files, desc="images"):
        stem = os.path.splitext(fn)[0]
        _, binary = predict_binary_mask(os.path.join(image_dir, fn), model, device, threshold)
        instances = extract_instances(binary, min_size, closing_kernel)
        total = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
        for k, m in enumerate(instances, start=1):
            rle = encode_mask(m)
            decode_bad += int(not (decode_rle(rle) == m).all())
            total += m
            rows.append((f"{stem}_{k}", rle))
        overlap_bad += int(total.max() > 1) if len(instances) else 0
        per_image[stem] = len(instances)
    write_submission(rows, out_csv)

    ids = [r[0] for r in rows]
    counts = np.array(list(per_image.values())) if per_image else np.zeros(1)
    return {
        "images": len(files),
        "rows": len(rows),
        "unique_ids": len(set(ids)) == len(ids),
        "no_quotes": all(chr(34) not in s and chr(39) not in s for _, s in rows),
        "images_with_overlapping_masks": overlap_bad,
        "rows_failing_rle_roundtrip": decode_bad,
        "mean_filaments_per_image": float(counts.mean()),
        "images_without_rows": sum(1 for v in per_image.values() if v == 0),
    }
