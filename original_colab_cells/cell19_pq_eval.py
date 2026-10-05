"""
Cell 19: ประเมิน Panoptic Quality (PQ) จาก mask ที่เซฟไว้ (CPU ล้วน ไม่ใช้ GPU)
================================================================================
- จับคู่ instance ที่ทำนาย กับ instance จริง (แต่ละ annotation) ด้วย IoU > 0.5
- PQ = SQ x RQ  โดย  RQ = TP / (TP + 0.5 FP + 0.5 FN),  SQ = ผลรวม IoU ของคู่ที่จับได้ / TP
- คอมโบที่เทียบ (กำหนดล่วงหน้า 5 แบบ ไม่ไล่กวาด เพื่อไม่ให้จูนเข้ากับ val set มากเกินไป)
- สำรอง mask ลง Google Drive เป็นไฟล์บีบอัดไฟล์เดียว และกู้คืนได้ถ้าเซสชันหลุด

ต้องรัน cell14_watershed.py ก่อน (ใช้ label_naive, label_watershed)
หมายเหตุ: ค่านี้เป็นค่าประมาณบน val ไม่ใช่คะแนน leaderboard จริง
(Kaggle มีวิธีตรวจสอบเพิ่มเติมที่เราไม่เห็นรายละเอียด และไม่ทราบว่าเฉลี่ยแบบรายภาพหรือรวมทั้งชุด
เลยรายงานทั้งสองแบบ)
"""

import os
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------
# ฟังก์ชันหลัก (ไม่พึ่งตัวแปรภายนอก)
# ---------------------------------------------------------
def filter_small_labels(labeled, min_size):
    """ตัด instance ที่เล็กกว่า min_size ทิ้ง แล้วรันเลข label ใหม่ให้ต่อเนื่อง 1..n"""
    n = int(labeled.max())
    if n == 0:
        return labeled, 0
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    keep = np.flatnonzero(sizes[1:] >= min_size) + 1
    lut = np.zeros(n + 1, dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return lut[labeled], len(keep)


def gt_pixel_lists(img_id, coco):
    """เฉลยรายเส้น: เก็บเป็นรายการตำแหน่งพิกเซล (ประหยัดหน่วยความจำกว่าเก็บ mask เต็ม)"""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    out = []
    for ann in anns:
        idx = np.flatnonzero(coco.annToMask(ann))
        if idx.size > 0:
            out.append(idx)
    return out


def match_instances(pred_labeled, gt_lists, iou_thresh=0.5):
    """จับคู่แบบ one-to-one (greedy ตาม IoU สูงสุดก่อน) คืน (TP, ผลรวม IoU, จำนวน pred, จำนวน GT)"""
    n_pred = int(pred_labeled.max())
    n_gt = len(gt_lists)
    if n_pred == 0 or n_gt == 0:
        return 0, 0.0, n_pred, n_gt

    pred_flat = pred_labeled.ravel()
    pred_area = np.bincount(pred_flat, minlength=n_pred + 1)[1:]

    candidates = []
    for g_i, idx in enumerate(gt_lists):
        inter = np.bincount(pred_flat[idx], minlength=n_pred + 1)[1:]
        union = pred_area + idx.size - inter
        iou = inter / np.maximum(union, 1)
        for p_i in np.flatnonzero(iou > iou_thresh):
            candidates.append((float(iou[p_i]), int(p_i), g_i))

    candidates.sort(reverse=True)
    used_p, used_g = set(), set()
    tp, sum_iou = 0, 0.0
    for iou, p_i, g_i in candidates:
        if p_i in used_p or g_i in used_g:
            continue
        used_p.add(p_i)
        used_g.add(g_i)
        tp += 1
        sum_iou += iou
    return tp, sum_iou, n_pred, n_gt


def pq_of(tp, sum_iou, n_pred, n_gt):
    fp = n_pred - tp
    fn = n_gt - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    if denom == 0:
        return None
    return sum_iou / denom


# ===== RUNNER =====
DRIVE_DIR_ = globals().get("DRIVE_DIR", "/content/drive/MyDrive/Colab Filament")
CACHE_DIR = "mask_cache_npy"
NPZ_BACKUP = os.path.join(DRIVE_DIR_, "mask_cache_val.npz")
os.makedirs(CACHE_DIR, exist_ok=True)

# 1) ให้มี coco (ถ้าเซสชันใหม่และยังไม่มีไฟล์ JSON ให้ copy จาก Drive มาก่อน)
if "coco" not in globals():
    import shutil
    from pycocotools.coco import COCO
    json_name_ = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not os.path.exists(json_name_):
        shutil.copy(os.path.join(DRIVE_DIR_, json_name_), ".")
    coco = COCO(json_name_)

# 2) ถ้าโฟลเดอร์ mask ว่าง แต่มีไฟล์สำรองใน Drive ให้กู้คืนก่อน
if len(os.listdir(CACHE_DIR)) == 0 and os.path.exists(NPZ_BACKUP):
    print("♻️ พบไฟล์สำรองใน Drive กำลังกู้คืน mask...")
    data = np.load(NPZ_BACKUP)
    for k in data.files:
        np.save(os.path.join(CACHE_DIR, k + ".npy"), data[k])
    print(f"✅ กู้คืน {len(data.files)} mask แล้ว")

# 3) ให้มี cached_info (สร้างจากไฟล์ที่มีอยู่ถ้ายังไม่มีในเซสชัน)
if "cached_info" not in globals():
    cached_info = []
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.endswith(".npy"):
            img_id = f[:-4]
            n_gt = len(coco.getAnnIds(imgIds=[img_id]))
            cached_info.append((img_id, os.path.join(CACHE_DIR, f), n_gt))
    print(f"📂 สร้าง cached_info จากไฟล์ที่มี: {len(cached_info)} ภาพ")

# 4) สำรอง mask ลง Drive (ทำครั้งเดียว)
if not os.path.exists(NPZ_BACKUP):
    arrays = {img_id: np.load(path) for img_id, path, _ in cached_info}
    np.savez_compressed(NPZ_BACKUP, **arrays)
    print(f"💾 สำรอง mask ลง Drive แล้ว: {NPZ_BACKUP} ({os.path.getsize(NPZ_BACKUP)/1e6:.1f} MB)")

# 5) ประเมิน PQ
CONFIGS = {
    "Naive":               ("naive", None, None),
    "Naive + size80":      ("naive", None, 80),
    "Watershed md=140":    ("ws", 140, None),
    "Watershed md=140 + size120": ("ws", 140, 120),
    "Watershed md=80 + size80":   ("ws", 80, 80),
}

stats = {name: {"tp": 0, "iou": 0.0, "pred": 0, "gt": 0, "pq_imgs": []} for name in CONFIGS}

for img_id, cache_path, _ in tqdm(cached_info, desc="ประเมิน PQ"):
    binary_mask = np.load(cache_path)
    gts = gt_pixel_lists(img_id, coco)
    if len(gts) == 0:
        continue

    # คำนวณ labeling แต่ละแบบครั้งเดียว แล้วใช้ซ้ำข้ามคอมโบ
    lab_naive, _ = label_naive(binary_mask)
    lab_ws140, _ = label_watershed(binary_mask, min_distance=140)
    lab_ws80, _ = label_watershed(binary_mask, min_distance=80)
    base = {("naive", None): lab_naive, ("ws", 140): lab_ws140, ("ws", 80): lab_ws80}

    for name, (method, md, min_size) in CONFIGS.items():
        labeled = base[(method, md)]
        if min_size is not None:
            labeled, _ = filter_small_labels(labeled.astype(np.int32), min_size)
        tp, s_iou, n_p, n_g = match_instances(labeled, gts)
        st = stats[name]
        st["tp"] += tp
        st["iou"] += s_iou
        st["pred"] += n_p
        st["gt"] += n_g
        pq_img = pq_of(tp, s_iou, n_p, n_g)
        if pq_img is not None:
            st["pq_imgs"].append(pq_img)

print(f"\n📊 === PQ บน val ({len(cached_info)} ภาพ) ===")
print(f"{'คอมโบ':30s} | PQ รวม | SQ    | RQ    | PQ เฉลี่ยรายภาพ | instance ที่ทำนาย/ภาพ")
for name, st in stats.items():
    tp, s_iou, n_p, n_g = st["tp"], st["iou"], st["pred"], st["gt"]
    pq = pq_of(tp, s_iou, n_p, n_g)
    sq = s_iou / tp if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * (n_p - tp) + 0.5 * (n_g - tp))
    pq_mean = float(np.mean(st["pq_imgs"]))
    print(f"{name:30s} | {pq:.4f} | {sq:.3f} | {rq:.3f} | {pq_mean:.4f}          | {n_p / len(cached_info):.2f}")
