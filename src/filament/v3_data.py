"""Training tiles of the supplementary experiments (numpy only, no torch).

Model B ("v3_control") and the other fine-tuning runs start from the baseline weights and are trained on:
  * all filament tiles + a seeded 25 % of the empty tiles of the training images,
  * minus 10 whole annotator-images (code B: ground truth marked, nothing visible),
  * minus 173 tiles (held-out A/C/D tiles and B* tiles that are not in the feather list),
  * the 448 hand-picked "feather" tiles (thin, feather-like filaments), repeated `feather_repeat` times.
Tile numbers refer to the frozen order of the training annotator-images:
    tile number = 20000 + 16 * index_in_train_ids + 4 * row + column      (512 px tiles, 4 x 4 per image)
so `train_ids` must come from `filament.data.split_by_date` and must never be re-ordered.
The lists are stored in experiments/tile_plan.json.
"""
import collections
import json
import os
import random

import cv2
import numpy as np

from .config import EMPTY_KEEP_RATIO, SPLIT_SEED, TILE_SIZE

SIZE = 2048
TILES_PER_SIDE = SIZE // TILE_SIZE          # 4
TILE_BASE = 20000


def load_plan(path):
    """Read experiments/tile_plan.json -> dict of sets."""
    with open(path) as f:
        plan = json.load(f)
    return {
        "feather": set(plan["feather_tiles"]),
        "exclude": set(plan["holdout_ACD_tiles"]) | set(plan["removed_Bstar_tiles"]),
        "drop_images": set(plan["dropped_images"]),
    }


def photo_of(img_id):
    """"010401-20141106195854Bh" -> "20141106195854Bh" (the photo shared by all annotators)."""
    return str(img_id).split("-", 1)[1]


def inst_masks(coco, img_id):
    return [coco.annToMask(a).astype(bool) for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))]


def union(masks):
    out = np.zeros((SIZE, SIZE), bool)
    for m in masks:
        out |= m
    return out


def make_unified(coco, by_photo, img_id, own_inst):
    """"Blue wins" label: filaments of the OTHER annotators of the same photo are merged (blue);
    where blue overlaps an owner filament the blue outline is used; owner-only filaments are kept;
    blue-only filaments are added."""
    blue = union([m for o in by_photo[photo_of(img_id)] if o != img_id for m in inst_masks(coco, o)])
    out = blue.copy()
    for g in own_inst:
        if not (g & blue).any():
            out |= g
    return out


def read_gray(coco, img_id, img_dir):
    info = coco.loadImgs([img_id])[0]
    return cv2.imread(os.path.join(img_dir, os.path.basename(info["file_name"])))[:, :, 0]


def build_training_tiles(coco, train_ids, plan, img_dir, feather_repeat=1, blue_wins=False,
                         empty_keep=EMPTY_KEEP_RATIO, seed=SPLIT_SEED):
    """Returns (imgs, masks, items, stats).

    imgs / masks : lists of (512, 512) uint8 arrays (grayscale image, 0/1 mask)
    items        : indices into imgs/masks for one epoch (feather tiles appear `feather_repeat` times)
    """
    rng = random.Random(seed)
    by_photo = collections.defaultdict(list)
    for i in coco.getImgIds():
        by_photo[photo_of(i)].append(i)

    imgs, masks, items = [], [], []
    stats = collections.Counter()
    T = TILE_SIZE
    for k, img_id in enumerate(train_ids):
        if img_id in plan["drop_images"]:
            continue
        own = own_inst = unified = img = None
        for j in range(TILES_PER_SIDE ** 2):
            n = TILE_BASE + 16 * k + j
            if n in plan["exclude"]:
                stats["excluded"] += 1
                continue
            is_f = n in plan["feather"]
            if own is None:
                own_inst = inst_masks(coco, img_id)
                own = union(own_inst)
            lab = own
            if is_f and blue_wins:
                if unified is None:
                    unified = make_unified(coco, by_photo, img_id, own_inst)
                lab = unified
            y, x = (j // TILES_PER_SIDE) * T, (j % TILES_PER_SIDE) * T
            m = lab[y:y + T, x:x + T]
            if is_f and (m != own[y:y + T, x:x + T]).any():
                stats["feather_label_changed"] += 1
            if not is_f and not m.any() and rng.random() >= empty_keep:
                stats["empty_dropped"] += 1
                continue
            if img is None:
                img = read_gray(coco, img_id, img_dir)
            imgs.append(img[y:y + T, x:x + T].copy())
            masks.append(m.astype(np.uint8))
            items += [len(imgs) - 1] * (feather_repeat if is_f else 1)
            stats["feather" if is_f else ("filament" if m.any() else "empty_kept")] += 1
    return imgs, masks, items, stats
