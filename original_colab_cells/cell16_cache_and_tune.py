"""
Cell 16: เซฟ mask ไว้ก่อน แล้วทดลองพารามิเตอร์ต่างๆ โดยไม่ต้องเรียก GPU ซ้ำ
================================================================================
ส่วนที่ 1 (ใช้ GPU ครั้งเดียว): ทำนายและเซฟ binary mask ของทุกภาพ val ไว้ในดิสก์
ส่วนที่ 2 (CPU ล้วนๆ): ทดลอง min_distance หลายค่า + area-filtering โดยไม่แตะ GPU อีกเลย
"""

import os
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

CACHE_DIR = "mask_cache_npy"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------
# ส่วนที่ 1: ทำนาย + เซฟ (รันแค่ครั้งเดียว ข้ามถ้ามีไฟล์แล้ว)
# ---------------------------------------------------------
THRESHOLD = 0.5
cached_info = []

print("🚀 กำลังทำนายและเซฟ mask ของทุกภาพ (ใช้ GPU ครั้งเดียว จบแล้วไม่ต้องเรียกอีก)...")
for img_id in tqdm(val_ids, desc="Predict & Cache"):
    info = coco.loadImgs([img_id])[0]
    path = os.path.join(IMG_DIR, os.path.basename(info['file_name']))
    if not os.path.exists(path):
        continue

    cache_path = os.path.join(CACHE_DIR, f"{img_id.replace('/', '_')}.npy")
    gt = get_full_gt(img_id, coco)
    if gt.sum() == 0:
        continue

    if not os.path.exists(cache_path):
        img_tensor = load_full_image_tensor(img_id, coco)
        _, binary_mask = predict_full_image_overlap(img_tensor, model, device, threshold=THRESHOLD)
        np.save(cache_path, binary_mask.astype(np.uint8))

    n_gt = count_gt_instances(img_id, coco)
    cached_info.append((img_id, cache_path, n_gt))

print(f"✅ เซฟ mask ไว้ที่ {CACHE_DIR}/ แล้ว {len(cached_info)} ไฟล์ (ใช้ซ้ำได้โดยไม่ต้องเรียก GPU อีก)")


# ---------------------------------------------------------
# ส่วนที่ 2: ฟังก์ชันทดลอง (CPU ล้วนๆ ไม่แตะโมเดล/GPU เลย)
# ---------------------------------------------------------
def remove_small_objects(binary_mask, min_size=20):
    """กรอง component ที่เล็กเกินไปทิ้ง (น่าจะเป็น noise ไม่ใช่ filament จริง)"""
    labeled, n = ndi.label(binary_mask)
    sizes = ndi.sum(binary_mask, labeled, range(1, n + 1))
    mask_out = np.zeros_like(binary_mask)
    for i, size in enumerate(sizes, start=1):
        if size >= min_size:
            mask_out[labeled == i] = 1
    return mask_out


def evaluate_setting(cached_info, method="naive", min_distance=15, min_size=None):
    errors = []
    counts = []
    for img_id, cache_path, n_gt in cached_info:
        binary_mask = np.load(cache_path)

        if min_size is not None:
            binary_mask = remove_small_objects(binary_mask, min_size=min_size)

        if method == "naive":
            _, n_pred = label_naive(binary_mask)
        elif method == "watershed":
            _, n_pred = label_watershed(binary_mask, min_distance=min_distance)

        counts.append(n_pred)
        errors.append(abs(n_pred - n_gt))

    return np.mean(counts), np.mean(errors)


# ---------------------------------------------------------
# ทดลองหลายค่า (ทั้งหมดนี้ไม่ใช้ GPU เลย เร็วมาก)
# ---------------------------------------------------------
gt_mean = np.mean([g for _, _, g in cached_info])
print(f"\nGT เฉลี่ย: {gt_mean:.2f}")
print("\n--- ทดลอง Naive + Area Filtering (ไม่ใช้ Watershed) ---")
for min_size in [None, 10, 20, 40, 80]:
    mean_count, mean_err = evaluate_setting(cached_info, method="naive", min_size=min_size)
    label = f"min_size={min_size}" if min_size else "ไม่กรอง (baseline)"
    print(f"  {label:20s} | เฉลี่ย instance = {mean_count:.2f} | คลาดเคลื่อนเฉลี่ย = {mean_err:.2f}")

print("\n--- ทดลอง Watershed ด้วย min_distance ที่ใหญ่ขึ้น ---")
for md in [15, 30, 50, 80, 120]:
    mean_count, mean_err = evaluate_setting(cached_info, method="watershed", min_distance=md)
    print(f"  min_distance={md:4d} | เฉลี่ย instance = {mean_count:.2f} | คลาดเคลื่อนเฉลี่ย = {mean_err:.2f}")
