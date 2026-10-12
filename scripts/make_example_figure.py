"""Create fig_examples.png for the report: raw crop and overlay for randomly drawn validation photos.

  python scripts/make_example_figure.py --json MAGFiLO_1.0_Annotations_kaggle2026_train.json --images train \
         --weights unet_resnet34_v3_control_ep5.pth --out fig_examples.png
Photos are drawn with random.Random(--seed) from the validation photos (no hand picking); the 768 px window is centred
on the centre of mass of all annotators masks. Cyan = every annotator of the photo, red = pieces of the final pipeline.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

from filament.config import MIN_SIZE, CLOSING_KERNEL  # noqa: E402
from filament.data import split_by_date  # noqa: E402
from filament.infer import predict_binary_mask  # noqa: E402
from filament.model import load_checkpoint  # noqa: E402
from filament.postprocess import extract_instances  # noqa: E402
from filament.v3_data import inst_masks, photo_of  # noqa: E402
from filament.viz import crop_box, overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="fig_examples.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2, help="number of photos")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = COCO(a.json)
    _, val_ids = split_by_date(coco.dataset["images"])
    by_photo = {}
    for i in val_ids:
        by_photo.setdefault(photo_of(i), []).append(i)
    photos = sorted(by_photo)
    chosen = random.Random(a.seed).sample(photos, a.n)
    print("chosen photos:", chosen)
    model = load_checkpoint(a.weights, device)

    fig, axes = plt.subplots(a.n, 2, figsize=(4.6, 2.3 * a.n))
    axes = np.atleast_2d(axes)
    for r, photo in enumerate(chosen):
        ids = by_photo[photo]
        fn = os.path.basename(coco.loadImgs([ids[0]])[0]["file_name"])
        path = os.path.join(a.images, fn)
        _, binary = predict_binary_mask(path, model, device)
        pred = extract_instances(binary, MIN_SIZE, CLOSING_KERNEL)
        gt = [m for i in ids for m in inst_masks(coco, i)]
        gt_union = np.zeros((2048, 2048), bool)
        for m in gt:
            gt_union |= m
        y0, x0 = crop_box(gt_union)
        sl = (slice(y0, y0 + 768), slice(x0, x0 + 768))
        gray = cv2.imread(path)[:, :, 0][sl]
        crop = lambda ms: [m[sl] for m in ms if m[sl].any()]
        axes[r, 0].imshow(gray, cmap="gray")
        axes[r, 1].imshow(overlay(gray, crop(gt), crop(pred)))
        axes[r, 0].set_title(photo, fontsize=6)
        axes[r, 1].set_title(f"{len(ids)} annotator(s); {len(crop(pred))} predicted piece(s) in crop", fontsize=6)
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(a.out, dpi=250)
    print("saved", a.out)


if __name__ == "__main__":
    main()
