"""Training-data preparation: date-grouped split and 512 px tiling (no torch needed)."""
import json
import os
import random

import cv2
import numpy as np

from .config import EMPTY_KEEP_RATIO, SPLIT_SEED, TILE_SIZE, VAL_RATIO


def split_by_date(images, seed=SPLIT_SEED, val_ratio=VAL_RATIO):
    """Group-by-date train/validation split.

    images : iterable of dicts with keys "id" and "file_name" (COCO "images" list).
    All annotator-images whose file name starts with the same YYYYMMDD go to the
    same side, so near-duplicate scenes cannot leak from train into validation.
    Returns (train_ids, val_ids) of *annotator-image ids*.
    """
    date_to_ids = {}
    for img in images:
        date_to_ids.setdefault(img["file_name"][:8], []).append(img["id"])
    dates = sorted(date_to_ids)
    random.Random(seed).shuffle(dates)
    val_dates = set(dates[:int(len(dates) * val_ratio)])
    train_ids, val_ids = [], []
    for date, ids in date_to_ids.items():
        (val_ids if date in val_dates else train_ids).extend(ids)
    return train_ids, val_ids


def union_mask(coco, img_id, height, width):
    """Union of all filament polygons of one annotator-image (uint8, 0/1)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
        mask = np.maximum(mask, coco.annToMask(ann))
    return mask


def build_tiles(coco, img_ids, split_name, img_dir, out_dir="tiles",
                tile=TILE_SIZE, empty_keep_ratio=EMPTY_KEEP_RATIO, empty_seed=0):
    """Cut images into non-overlapping tiles and save them as PNG.

    train: every tile with filament pixels + a random `empty_keep_ratio` of the empty tiles
    val  : ALL tiles (validation must reflect the real class balance)
    Tile files are named after the unique annotator-image id (never the file name),
    otherwise two annotators of the same photo would overwrite each other.
    NOTE: the original run did NOT seed the empty-tile sampling; `empty_seed` makes it
    reproducible from now on, but a re-cut will not equal the tile set of the original run.
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    rng = random.Random(empty_seed)
    manifest = []
    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        image = cv2.imread(os.path.join(img_dir, os.path.basename(info["file_name"])))
        if image is None:
            continue
        h, w = image.shape[:2]
        mask = union_mask(coco, img_id, h, w)
        base = str(img_id).replace("/", "_").replace(" ", "_")
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                tm = mask[y:y + tile, x:x + tile]
                if tm.shape != (tile, tile):
                    continue
                has = tm.sum() > 0
                if split_name == "val" or has or rng.random() < empty_keep_ratio:
                    name = f"{base}_y{y}_x{x}.png"
                    cv2.imwrite(os.path.join(out_dir, "images", name), image[y:y + tile, x:x + tile])
                    cv2.imwrite(os.path.join(out_dir, "masks", name), tm * 255)
                    manifest.append(name)
    with open(os.path.join(out_dir, f"{split_name}_manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest
