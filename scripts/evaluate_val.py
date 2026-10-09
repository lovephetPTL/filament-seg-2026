"""Validation PQ of a checkpoint with the repository pipeline (same matching rule as the organisers code).

  python scripts/evaluate_val.py --json MAGFiLO_1.0_Annotations_kaggle2026_train.json --images train \
         --weights unet_resnet34_v3_control_ep5.pth --closing 31 --out val_v3_control_closing31.json
Use --closing 0 for plain connected components. Sanity targets measured earlier with the baseline weights:
  --closing 0  -> PQ 0.3825 (TP 774, FP 818, FN 381)     --closing 31 -> PQ 0.3937
This script has NOT been run yet: run it once on the baseline and check these two numbers first.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from tqdm import tqdm  # noqa: E402

from filament.config import MIN_SIZE  # noqa: E402
from filament.data import split_by_date  # noqa: E402
from filament.evaluation import evaluate_pq  # noqa: E402
from filament.infer import predict_binary_mask  # noqa: E402
from filament.model import load_checkpoint  # noqa: E402
from filament.postprocess import extract_instances  # noqa: E402
from filament.v3_data import photo_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--closing", type=int, default=31, help="0 = plain connected components")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = COCO(a.json)
    _, val_ids = split_by_date(coco.dataset["images"])
    model = load_checkpoint(a.weights, device)

    first_of_photo = {}
    for i in val_ids:
        first_of_photo.setdefault(photo_of(i), i)
    instances_by_photo = {}
    for photo, img_id in tqdm(first_of_photo.items(), desc="predict validation photos"):
        fn = os.path.basename(coco.loadImgs([img_id])[0]["file_name"])
        _, binary = predict_binary_mask(os.path.join(a.images, fn), model, device)
        instances_by_photo[photo] = extract_instances(binary, MIN_SIZE, a.closing or None)

    result = evaluate_pq(coco, val_ids, instances_by_photo, photo_of)
    print(json.dumps(result, indent=1))
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
