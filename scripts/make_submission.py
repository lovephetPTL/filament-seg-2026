"""Create the Kaggle submission CSV from a folder of test images.

  python scripts/make_submission.py --images test_images --weights best_baseline_unet.pth \
         --out submission_closing31B.csv                # final file (public score 0.32)
  ... --no-closing --out submission_baseline.csv        # baseline file (public score 0.31)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from filament.config import CLOSING_KERNEL  # noqa: E402
from filament.model import load_checkpoint  # noqa: E402
from filament.pipeline import make_submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with the 180 test images (flat)")
    ap.add_argument("--weights", default="best_baseline_unet.pth")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--no-closing", action="store_true", help="baseline: plain connected components")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "(CPU works but is slow: 25 tiles x 180 images)")
    model = load_checkpoint(a.weights, device)
    report = make_submission(a.images, model, device, a.out,
                             closing_kernel=None if a.no_closing else CLOSING_KERNEL)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
