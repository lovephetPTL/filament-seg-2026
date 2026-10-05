"""PyTorch dataset of pre-cut tiles."""
import os

import cv2
import torch
from torch.utils.data import Dataset

from .dataset_utils import normalize_np, read_rgb


class SolarTiledDataset(Dataset):
    def __init__(self, manifest, tiles_dir="tiles"):
        self.manifest, self.tiles_dir = manifest, tiles_dir

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        name = self.manifest[idx]
        image = normalize_np(read_rgb(os.path.join(self.tiles_dir, "images", name)))
        mask = cv2.imread(os.path.join(self.tiles_dir, "masks", name), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(image), torch.from_numpy(mask).float().unsqueeze(0) / 255.0
