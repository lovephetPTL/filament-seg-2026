"""
Cell: GPU Full Evaluation (Naive Grid vs Overlap) - ทำทุกอย่างในเซลล์เดียว ประหยัดเวลา GPU
================================================================================
ก่อนรัน: ต้อง mount Google Drive และอัปโหลด best_baseline_unet.pth เข้า Colab ก่อน
(ถ้ายังไม่ได้ mount ให้รัน `from google.colab import drive; drive.mount('/content/drive')` ก่อน)
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


# ---------------------------------------------------------
# รันประเมินผลเต็ม val set (164 ภาพ) ด้วย GPU
# ---------------------------------------------------------
THRESHOLD = 0.5
results = []

print(f"🚀 กำลังประเมินผลเต็ม val set ({len(val_ids)} ภาพ) ด้วย GPU...")
for img_id in tqdm(val_ids, desc="Naive vs Overlap (Full Val, GPU)"):
    info = coco.loadImgs([img_id])[0]
    path = os.path.join(IMG_DIR, os.path.basename(info['file_name']))
    if not os.path.exists(path):
        continue

    img_tensor = load_full_image_tensor(img_id, coco)
    gt = get_full_gt(img_id, coco)
    if gt.sum() == 0:
        continue

    _, naive_mask = predict_naive_grid(img_tensor, model, device, threshold=THRESHOLD)
    _, overlap_mask = predict_full_image_overlap(img_tensor, model, device, threshold=THRESHOLD)

    dice_naive = dice_np(naive_mask, gt)
    dice_overlap = dice_np(overlap_mask, gt)
    results.append((img_id, dice_naive, dice_overlap))

dice_naive_all = np.array([r[1] for r in results])
dice_overlap_all = np.array([r[2] for r in results])
improved = sum(1 for r in results if r[2] > r[1] + 1e-6)

print(f"\n📊 === ผลลัพธ์เต็ม val set ({len(results)} ภาพที่มี filament) ===")
print(f"Dice เฉลี่ย Naive Grid:      {dice_naive_all.mean():.4f}")
print(f"Dice เฉลี่ย Overlap:         {dice_overlap_all.mean():.4f}")
print(f"ภาพที่ดีขึ้นเมื่อใช้ overlap: {improved}/{len(results)} ({improved/len(results)*100:.1f}%)")

# เซฟผลไว้กันเซสชันหลุด (เผื่ออยากดูรายละเอียดทีหลัง)
with open("overlap_vs_naive_results.json", "w") as f:
    json.dump(results, f)
print("💾 บันทึกผลลัพธ์ไว้ที่ overlap_vs_naive_results.json แล้ว")

# 🎯 สำรองไฟล์ผลลัพธ์ลง Google Drive กันเซสชันหลุดแล้วข้อมูลหาย
import shutil
shutil.copy("overlap_vs_naive_results.json", os.path.join(DRIVE_DIR, "overlap_vs_naive_results.json"))
print("💾 สำรองไฟล์ผลลัพธ์ลง Google Drive เรียบร้อย!")
