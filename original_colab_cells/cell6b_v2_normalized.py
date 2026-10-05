"""
Cell 6b (v2): SolarTiledDataset + ImageNet Normalization
============================================================
เพิ่ม normalize ตามค่ามาตรฐานของ ImageNet เพราะ encoder ResNet34
ถูกโหลดมาจาก pretrained weights ที่เทรนบนภาพ normalize แบบนี้
"""

import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SolarTiledDataset(Dataset):
    def __init__(self, manifest, tiles_dir="tiles"):
        self.manifest = manifest
        self.tiles_dir = tiles_dir

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        tile_name = self.manifest[idx]
        img_path = os.path.join(self.tiles_dir, "images", tile_name)
        mask_path = os.path.join(self.tiles_dir, "masks", tile_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD  # 🎯 normalize แบบ ImageNet

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, mask


with open("tiles/train_manifest.json") as f:
    train_manifest = json.load(f)
with open("tiles/val_manifest.json") as f:
    val_manifest = json.load(f)

train_dataset = SolarTiledDataset(train_manifest)
val_dataset = SolarTiledDataset(val_manifest)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"✅ Dataset พร้อมใช้งาน (มี ImageNet normalize แล้ว): train {len(train_dataset)} tiles, val {len(val_dataset)} tiles")
