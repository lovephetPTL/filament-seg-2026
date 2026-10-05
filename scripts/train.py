"""Optional: prepare tiles and train. NOT needed to reproduce the submission.

Example (one session, like session 3 of the README history):
  python scripts/train.py --json MAGFiLO_1.0_Annotations_kaggle2026_train.json \
      --images train --start-epoch 11 --end-epoch 18 --lr 1e-4 --select-by dice \
      --resume best_baseline_unet.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from filament.data import build_tiles, split_by_date  # noqa: E402
from filament.dataset import SolarTiledDataset  # noqa: E402
from filament.model import build_model  # noqa: E402
from filament.train import fit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--tiles", default="tiles")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--start-epoch", type=int, default=1)
    ap.add_argument("--end-epoch", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--select-by", choices=["loss", "dice"], default="loss")
    ap.add_argument("--initial-best-loss", type=float, default=float("inf"))
    ap.add_argument("--save", default="best_baseline_unet.pth")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = COCO(a.json)
    train_ids, val_ids = split_by_date(coco.dataset["images"])
    if not os.path.exists(os.path.join(a.tiles, "train_manifest.json")):
        build_tiles(coco, train_ids, "train", a.images, a.tiles)
        build_tiles(coco, val_ids, "val", a.images, a.tiles)
    tr = json.load(open(os.path.join(a.tiles, "train_manifest.json")))
    va = json.load(open(os.path.join(a.tiles, "val_manifest.json")))
    train_loader = DataLoader(SolarTiledDataset(tr, a.tiles), batch_size=a.batch_size, shuffle=True)
    val_loader = DataLoader(SolarTiledDataset(va, a.tiles), batch_size=a.batch_size, shuffle=False)

    model = build_model(pretrained_encoder=a.resume is None).to(device)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=device))
    fit(model, train_loader, val_loader, device, a.start_epoch, a.end_epoch, a.lr,
        select_by=a.select_by, save_path=a.save, initial_best_loss=a.initial_best_loss)


if __name__ == "__main__":
    main()
