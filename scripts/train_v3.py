"""Supplementary experiments: continue training from the baseline weights on the cleaned tile set.

Model B ("v3_control", public 0.33 with closing) is the DEFAULT of this script:
  python scripts/train_v3.py --json MAGFiLO_1.0_Annotations_kaggle2026_train.json --images train \
      --start-weights unet_resnet34_baseline_ep18.pth --run-name exp_v3_control --out runs
Other arm of the same experiment ("v3_main", public 0.31): add  --feather-repeat 4 --blue-wins
Settings fixed before seeing any result: lr 1e-4, 5 epochs, batch 8, seed 42, random 90-degree rotation,
BatchNorm not frozen, AdamW weight decay 1e-2, loss 0.5 BCE + 0.5 Dice. Judge the LAST epoch only.
Colab T4: about 10 minutes per epoch (7,994 samples).
This script is a cleaned port of the Colab cell used for the experiments; it has NOT been re-run yet.
"""
import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402
import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from filament.config import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from filament.data import split_by_date  # noqa: E402
from filament.v3_data import build_training_tiles, load_plan  # noqa: E402

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


class TileSet(Dataset):
    def __init__(self, imgs, masks, items):
        self.imgs, self.masks, self.items = imgs, masks, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        j = self.items[i]
        r = int(torch.randint(0, 4, (1,)))               # random rotation 0/90/180/270, same for image and mask
        im = np.rot90(self.imgs[j], r).copy()
        mk = np.rot90(self.masks[j], r).copy()
        x = torch.from_numpy(im).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        return (x - MEAN) / STD, torch.from_numpy(mk).float().unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--start-weights", required=True, help="baseline checkpoint (14 epochs)")
    ap.add_argument("--plan", default=os.path.join(os.path.dirname(__file__), "..", "experiments", "tile_plan.json"))
    ap.add_argument("--run-name", default="exp_v3_control")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--feather-repeat", type=int, default=1)
    ap.add_argument("--blue-wins", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = os.path.join(a.out, a.run_name)
    assert not os.path.exists(out), f"{out} exists already - choose another --run-name (never overwrite a run)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    coco = COCO(a.json)
    train_ids, _ = split_by_date(coco.dataset["images"])
    assert len(train_ids) == 990, "tile numbers refer to the frozen order of the 990 training annotator-images"
    plan = load_plan(a.plan)
    imgs, masks, items, stats = build_training_tiles(
        coco, train_ids, plan, a.images, feather_repeat=a.feather_repeat, blue_wins=a.blue_wins, seed=a.seed)
    print("tile statistics:", dict(stats), "| samples per epoch:", len(items))

    loader = DataLoader(TileSet(imgs, masks, items), batch_size=a.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed), num_workers=0)
    net = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    net.load_state_dict(torch.load(a.start_weights, map_location=device))
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)

    os.makedirs(out)
    json.dump({"settings": vars(a), "stats": dict(stats), "samples_per_epoch": len(items)},
              open(os.path.join(out, "plan.json"), "w"), indent=1)
    with open(os.path.join(out, "log.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "minutes"])
    for ep in range(1, a.epochs + 1):
        t0, run, cnt = time.time(), 0.0, 0
        net.train()
        for x, y in tqdm(loader, desc=f"{a.run_name} epoch {ep}/{a.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = net(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice(logits, y)
            loss.backward()
            opt.step()
            run += loss.item() * len(x)
            cnt += len(x)
        torch.save(net.state_dict(), os.path.join(out, f"ep{ep}.pth"))
        with open(os.path.join(out, "log.csv"), "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{run / cnt:.4f}", f"{(time.time() - t0) / 60:.1f}"])
        print(f"epoch {ep}: train loss {run / cnt:.4f}")
    print("done. Judge ep%d.pth only ->" % a.epochs, out)


if __name__ == "__main__":
    main()
