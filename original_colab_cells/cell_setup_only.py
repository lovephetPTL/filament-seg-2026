"""
Cell: Setup Only (ตั้งค่าเซสชัน ไม่รันการประเมิน) ประหยัด GPU ~3 นาที
================================================================================
เหมือน cell_gpu_full_eval.py แต่ตัดส่วน "ประเมิน Naive vs Overlap 164 ภาพ" ออก
เหลือ: แตกไฟล์ภาพ, โหลด JSON/coco, แบ่ง train/val, โหลดโมเดล, นิยามฟังก์ชัน overlap inference
ก่อนรัน: mount Drive และ copy best_baseline_unet.pth มาไว้ใน /content ให้เรียบร้อย
"""
# ---------------------------------------------------------
# ติดตั้ง library ที่จำเป็น
# ---------------------------------------------------------
!pip install -q segmentation-models-pytorch pycocotools

import os
import cv2
import json
import random
import zipfile
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pycocotools.coco import COCO
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ กำลังใช้ Device: {device}")
assert device.type == "cuda", "⚠️ ยังไม่ได้ต่อ GPU! เช็ค Runtime > Change runtime type ก่อนรันต่อ"

DRIVE_DIR = "/content/drive/MyDrive/Colab Filament"  # แก้ path ถ้าคุณเก็บไว้ที่อื่น

# ---------------------------------------------------------
# แตกไฟล์ภาพ (ถ้ายังไม่มี /content/train)
# ---------------------------------------------------------
if not os.path.exists("train") or len(os.listdir("train")) < 100:
    print("📦 กำลังแตกไฟล์ภาพจาก Drive...")
    os.makedirs("train", exist_ok=True)
    zip_path = os.path.join(DRIVE_DIR, "train.zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if member.endswith("/") or not member.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            basename = os.path.basename(member)
            with z.open(member) as src, open(os.path.join("train", basename), "wb") as dst:
                dst.write(src.read())
    print(f"✅ แตกไฟล์แล้ว {len(os.listdir('train'))} ไฟล์")
else:
    print(f"✅ พบไฟล์ภาพอยู่แล้ว {len(os.listdir('train'))} ไฟล์ ข้ามการแตกไฟล์")

# copy JSON มาไว้ที่ /content/ (ถ้ายังไม่มี)
json_name = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
if not os.path.exists(json_name):
    import shutil
    shutil.copy(os.path.join(DRIVE_DIR, json_name), ".")
    print("✅ Copy JSON มาแล้ว")

# ---------------------------------------------------------
# โหลด coco + แบ่ง train/val (group by date เหมือนเดิม)
# ---------------------------------------------------------
coco = COCO(json_name)

date_to_img_ids = {}
for img in coco.dataset["images"]:
    date_key = img["file_name"][:8]
    date_to_img_ids.setdefault(date_key, []).append(img["id"])

unique_dates = sorted(date_to_img_ids.keys())
random.Random(42).shuffle(unique_dates)
n_val_dates = int(len(unique_dates) * 0.15)
val_dates = set(unique_dates[:n_val_dates])

train_ids, val_ids = [], []
for date_key, ids in date_to_img_ids.items():
    if date_key in val_dates:
        val_ids.extend(ids)
    else:
        train_ids.extend(ids)

print(f"train_ids: {len(train_ids)}, val_ids: {len(val_ids)}")

# ---------------------------------------------------------
# สร้างโมเดล + โหลดน้ำหนัก
# ---------------------------------------------------------
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
model.load_state_dict(torch.load("best_baseline_unet.pth", map_location=device))
model.eval()
print("✅ โหลดโมเดลสำเร็จ")

# ---------------------------------------------------------
# ฟังก์ชัน Overlap Inference (จาก cell12)
# ---------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TILE_SIZE = 512
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP
IMG_DIR = "train"


def create_2d_gaussian_window(size, sigma=0.5):
    grid = np.linspace(-1, 1, size)
    x, y = np.meshgrid(grid, grid)
    d = np.sqrt(x * x + y * y)
    return np.exp(-(d ** 2 / (2.0 * sigma ** 2)))


blend_weight = create_2d_gaussian_window(TILE_SIZE)


def load_full_image_tensor(img_id, coco, img_dir=IMG_DIR):
    info = coco.loadImgs([img_id])[0]
    path = os.path.join(img_dir, os.path.basename(info['file_name']))
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def get_full_gt(img_id, coco):
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))
    return mask


def dice_np(pred, gt, eps=1e-7):
    intersection = (pred * gt).sum()
    return (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)


def predict_naive_grid(image_tensor, model, device, threshold=0.5, tile_size=512):
    H, W = image_tensor.shape[-2:]
    coords = [(y, x) for y in range(0, H, tile_size) for x in range(0, W, tile_size)]
    batch = torch.stack([image_tensor[:, y:y + tile_size, x:x + tile_size] for y, x in coords]).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).cpu().squeeze(1).numpy()
    full_prob = np.zeros((H, W), dtype=np.float32)
    for (y, x), p in zip(coords, probs):
        full_prob[y:y + tile_size, x:x + tile_size] = p
    return full_prob, (full_prob > threshold).astype(np.float32)


def predict_full_image_overlap(image_2048, model, device, threshold=0.5):
    H, W = image_2048.shape[-2:]
    prob_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    y_steps = list(range(0, H - TILE_SIZE + 1, STRIDE))
    if y_steps[-1] + TILE_SIZE < H:
        y_steps.append(H - TILE_SIZE)
    x_steps = list(range(0, W - TILE_SIZE + 1, STRIDE))
    if x_steps[-1] + TILE_SIZE < W:
        x_steps.append(W - TILE_SIZE)

    coords = [(y, x) for y in y_steps for x in x_steps]
    tiles_batch = torch.stack([image_2048[:, y:y + TILE_SIZE, x:x + TILE_SIZE] for y, x in coords]).to(device)

    with torch.no_grad():
        tiles_prob = torch.sigmoid(model(tiles_batch)).cpu().squeeze(1).numpy()

    for (y, x), tile_prob in zip(coords, tiles_prob):
        prob_map[y:y + TILE_SIZE, x:x + TILE_SIZE] += tile_prob * blend_weight
        weight_map[y:y + TILE_SIZE, x:x + TILE_SIZE] += blend_weight

    final_prob_map = prob_map / (weight_map + 1e-7)
    return final_prob_map, (final_prob_map > threshold).astype(np.float32)


print("\n✅ ตั้งค่าเสร็จแล้ว: มี model, device, coco, val_ids, predict_full_image_overlap พร้อมใช้")
