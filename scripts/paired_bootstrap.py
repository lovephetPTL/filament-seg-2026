"""Paired bootstrap of PQ: Model B versus Model A on the validation photos (final pipeline, closing k=31).

  python scripts/paired_bootstrap.py --json MAGFiLO_1.0_Annotations_kaggle2026_train.json --images train \
      --weights-a unet_resnet34_baseline_ep18.pth --weights-b unet_resnet34_v3_control_ep5.pth --out paired_bootstrap.json
Sanity targets (measured earlier with the organisers code): PQ_A = 0.3937 and PQ_B = 0.3983.
Rule fixed before the result: the difference counts as supported only if the 95 % interval of B minus A excludes 0.
The interval covers the sampling of the 103 validation photos only, not the variance between training runs.
Run on a Colab T4 (about 2 min 20 s per model for the 103 validation photos): PQ_A 0.3937, PQ_B 0.3983, difference +0.0046,
95 % interval of B minus A [-0.0086, +0.0175] (10,000 resamples, seed 0), so Model B is not claimed to be better.
The result is stored in experiments/results/paired_bootstrap.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from tqdm import tqdm  # noqa: E402

from filament.bootstrap import paired_bootstrap  # noqa: E402
from filament.config import CLOSING_KERNEL, MIN_SIZE  # noqa: E402
from filament.data import split_by_date  # noqa: E402
from filament.evaluation import per_photo_counts  # noqa: E402
from filament.infer import predict_binary_mask  # noqa: E402
from filament.model import load_checkpoint  # noqa: E402
from filament.postprocess import extract_instances  # noqa: E402
from filament.v3_data import photo_of  # noqa: E402


def counts_for(weights, coco, val_ids, first_of_photo, images, device, closing):
    model = load_checkpoint(weights, device)
    inst = {}
    for photo, img_id in tqdm(first_of_photo.items(), desc=os.path.basename(weights)):
        fn = os.path.basename(coco.loadImgs([img_id])[0]["file_name"])
        _, binary = predict_binary_mask(os.path.join(images, fn), model, device)
        inst[photo] = extract_instances(binary, MIN_SIZE, closing or None)
    del model
    return per_photo_counts(coco, val_ids, inst, photo_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights-a", required=True)
    ap.add_argument("--weights-b", required=True)
    ap.add_argument("--closing", type=int, default=CLOSING_KERNEL)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paired_bootstrap.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = COCO(a.json)
    _, val_ids = split_by_date(coco.dataset["images"])
    first = {}
    for i in val_ids:
        first.setdefault(photo_of(i), i)
    ca = counts_for(a.weights_a, coco, val_ids, first, a.images, device, a.closing)
    cb = counts_for(a.weights_b, coco, val_ids, first, a.images, device, a.closing)
    photos = sorted(ca)
    res = paired_bootstrap(np.stack([ca[p] for p in photos]), np.stack([cb[p] for p in photos]), a.n_boot, a.seed)
    print(json.dumps(res, indent=1))
    if abs(res["PQ_A"] - 0.3937) > 0.0006 or abs(res["PQ_B"] - 0.3983) > 0.0006:
        print("WARNING: point estimates differ from the earlier 0.3937 / 0.3983 - check settings before using the interval")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
