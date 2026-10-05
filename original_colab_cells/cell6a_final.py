import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

# ---------------------------------------------------------
IMG_DIR = "train"
TILE_SIZE = 512
EMPTY_KEEP_RATIO = 0.25
VAL_RATIO = 0.15
SEED = 42
TILES_DIR = "tiles"

os.makedirs(f"{TILES_DIR}/images", exist_ok=True)
os.makedirs(f"{TILES_DIR}/masks", exist_ok=True)

coco = COCO("MAGFiLO_1.0_Annotations_kaggle2026_train.json")

# ---------------------------------------------------------
# 🎯 แบ่ง train/val แบบ "Group by Date" (ไม่ใช่สุ่มธรรมดา!)
# ป้องกัน data leakage เพราะภาพวันเดียวกันหน้าตาคล้ายกันมาก
# ---------------------------------------------------------
date_to_img_ids = {}
for img in coco.dataset["images"]:
    date_key = img["file_name"][:8]  # 8 หลักแรก = YYYYMMDD
    date_to_img_ids.setdefault(date_key, []).append(img["id"])

unique_dates = sorted(date_to_img_ids.keys())
random.Random(SEED).shuffle(unique_dates)

n_val_dates = int(len(unique_dates) * VAL_RATIO)
val_dates = set(unique_dates[:n_val_dates])

train_ids, val_ids = [], []
for date_key, ids in date_to_img_ids.items():
    if date_key in val_dates:
        val_ids.extend(ids)
    else:
        train_ids.extend(ids)

print(f"📅 จำนวนวันทั้งหมด: {len(unique_dates)}")
print(f"☀️ train: {len(train_ids)} ภาพ (image-level, ก่อนกรองไฟล์ที่หาไม่เจอ)")
print(f"🧪 val:   {len(val_ids)} ภาพ (image-level, ก่อนกรองไฟล์ที่หาไม่เจอ)")
print("-" * 50)


def process_and_save_tiles(img_ids, split_name):
    manifest = []
    filament_tiles_count = 0
    empty_tiles_count = 0
    skipped_imgs_count = 0

    print(f"🔄 กำลัง Precompute สำหรับชุด {split_name}...")

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        # 🎯 ใช้ img_id (ไม่ซ้ำกันเลย) ตั้งชื่อ tile แทน file_name (ซ้ำกันได้
        # เพราะภาพเดียวกันอาจถูกขีดเฉลยโดย annotator คนละคน คนละ id)
        # ป้องกันไฟล์ tile ของ annotator คนแรกถูกเขียนทับโดยคนหลัง
        base_name = str(img_id).replace("/", "_").replace(" ", "_")

        img_path = os.path.join(IMG_DIR, os.path.basename(file_name))
        image = cv2.imread(img_path)

        if image is None:
            skipped_imgs_count += 1
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))

        for y in range(0, h, TILE_SIZE):
            for x in range(0, w, TILE_SIZE):
                tile_mask = mask[y:y + TILE_SIZE, x:x + TILE_SIZE]

                if tile_mask.shape[0] != TILE_SIZE or tile_mask.shape[1] != TILE_SIZE:
                    continue  # tile ขอบภาพที่ไม่ครบขนาด ข้าม

                has_filament = np.sum(tile_mask) > 0

                keep_tile = False
                if split_name == "val":
                    keep_tile = True
                    if has_filament:
                        filament_tiles_count += 1
                    else:
                        empty_tiles_count += 1
                else:
                    if has_filament:
                        keep_tile = True
                        filament_tiles_count += 1
                    else:
                        if random.random() < EMPTY_KEEP_RATIO:
                            keep_tile = True
                            empty_tiles_count += 1

                if keep_tile:
                    tile_img = image[y:y + TILE_SIZE, x:x + TILE_SIZE]
                    tile_filename = f"{base_name}_y{y}_x{x}.png"
                    cv2.imwrite(os.path.join(TILES_DIR, "images", tile_filename), tile_img)
                    cv2.imwrite(os.path.join(TILES_DIR, "masks", tile_filename), tile_mask * 255)
                    manifest.append(tile_filename)

    manifest_path = f"{TILES_DIR}/{split_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\n✅ ประมวลผลชุด {split_name} สำเร็จ!")
    print(f"   - Tile มี filament: {filament_tiles_count:,} ชิ้น")
    print(f"   - Tile ว่างเปล่าที่เก็บไว้: {empty_tiles_count:,} ชิ้น")
    print(f"   - รวมทั้งหมดใน manifest: {len(manifest):,} ชิ้น")
    if skipped_imgs_count > 0:
        print(f"   ⚠️ หาไฟล์ภาพไม่เจอ: {skipped_imgs_count} ภาพ")
    print("-" * 50)


process_and_save_tiles(train_ids, "train")
process_and_save_tiles(val_ids, "val")
