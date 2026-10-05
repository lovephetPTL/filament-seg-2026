# cell24_closing_ablation.py
# ทดสอบ morphological closing (เชื่อมชิ้นที่แตก) เทียบกับ baseline บน val ด้วย PQ
# ใช้ mask ที่แคชไว้ (mask_cache_val.npz) จึงไม่ใช้ GPU
# baseline ต้องได้ TP=774 FP=818 PQ=0.3825 เท่ากับ cell23 ถ้าไม่ตรงให้หยุดตรวจก่อน
# ค่า k กำหนดล่วงหน้า 3 ค่า ห้ามไล่กวาดเพิ่มจากตารางนี้

import os, cv2, numpy as np
from collections import defaultdict
from scipy import ndimage as ndi
from tqdm import tqdm

DRIVE_DIR = "/content/drive/MyDrive/Colab Filament"
NPZ = os.path.join(DRIVE_DIR, "mask_cache_val.npz")
MIN_SIZE, IOU_T = 120, 0.5
CONFIGS = [("baseline", 0, None)]
for k in (7, 15, 31):
    CONFIGS += [(f"closing k={k} A(ส่งmaskที่ปิดแล้ว)", k, "A"),
                (f"closing k={k} B(ใช้จัดกลุ่มอย่างเดียว)", k, "B")]

data = np.load(NPZ)
by_file = defaultdict(list)
for i in val_ids:
    by_file[coco.loadImgs([i])[0]["file_name"]].append(i)

gts_of = {}
for i in val_ids:
    lst = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[i])):
        idx = np.flatnonzero(coco.annToMask(ann))
        if idx.size:
            lst.append(idx)
    gts_of[i] = lst


def make_instances(binary, closed, mode):
    if mode == "B":
        lab, n = ndi.label(closed)
        lab = lab * binary
    else:
        lab, n = ndi.label(binary if closed is None else closed)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.flatnonzero(sizes[1:] >= MIN_SIZE) + 1
    lut = np.zeros(n + 1, dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[lab].ravel(), len(keep), sizes[keep].astype(np.float64)


def match(lab2, K, area, gts, diag=False):
    cands, info = [], []
    for g, idx in enumerate(gts):
        inter = np.bincount(lab2[idx], minlength=K + 1)
        if K:
            iou = inter[1:] / (area + idx.size - inter[1:])
            for k in np.flatnonzero(iou > IOU_T):
                cands.append((float(iou[k]), int(k), g))
        if diag:
            info.append(((idx.size - inter[0]) / idx.size, int((inter[1:] > 0).sum())))
    cands.sort(reverse=True)
    uk, ug, tp, s = set(), set(), 0, 0.0
    for v, k, g in cands:
        if k in uk or g in ug:
            continue
        uk.add(k); ug.add(g); tp += 1; s += v
    un = [info[g] for g in range(len(gts)) if g not in ug] if diag else None
    return tp, s, K, len(gts), un


def pq(tp, s, npred, ng):
    d = tp + 0.5 * (npred - tp) + 0.5 * (ng - tp)
    return s / d if d else float("nan")


tot = {c[0]: np.zeros(4) for c in CONFIGS}
perfile = {c[0]: [] for c in CONFIGS}
diag_rows = []

for fname, ids in tqdm(by_file.items(), desc="val files"):
    binary = data[ids[0]].astype(np.uint8)
    closed = {0: None}
    for k in (7, 15, 31):
        closed[k] = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    for name, k, mode in CONFIGS:
        lab2, K, area = make_instances(binary, closed[k], mode)
        acc = np.zeros(4)
        for i in ids:
            if not gts_of[i]:
                continue
            tp, s, npred, ng, un = match(lab2, K, area, gts_of[i], diag=(name == "baseline"))
            acc += (tp, s, npred, ng)
            if un:
                diag_rows.extend(un)
        tot[name] += acc
        perfile[name].append(pq(acc[0], acc[1], acc[2], acc[3]))

print(f"\n{'วิธี':44s} | TP   FP   FN  | SQ    RQ    PQ     | ชิ้น/ไฟล์ | ไฟล์ดีขึ้น/แย่ลงเทียบ baseline")
base_pf = np.array(perfile["baseline"])
for name, k, mode in CONFIGS:
    tp, s, npred, ng = tot[name]
    rq = tp / (tp + 0.5 * (npred - tp) + 0.5 * (ng - tp))
    pf = np.array(perfile[name])
    better = int(np.sum(pf > base_pf + 1e-9)); worse = int(np.sum(pf < base_pf - 1e-9))
    print(f"{name:44s} | {int(tp)} {int(npred - tp)} {int(ng - tp)} | {s / tp:.3f} {rq:.3f} {pq(tp, s, npred, ng):.4f} | {npred / len(by_file):6.2f}    | {better}/{worse}")

# ตรวจสมมติฐาน "แตกเป็นท่อน" โดยตรงบน baseline
big = [r for r in diag_rows if r[0] >= 0.5]
print(f"\n--- baseline: FN ที่ถูกคลุม >=50% มี {len(big)} เส้น ---")
print(f"  ถูกคลุมด้วยชิ้นเดียว (รูปร่าง/ขอบไม่ตรง): {sum(r[1] == 1 for r in big)}")
print(f"  ถูกคลุมด้วย 2 ชิ้นขึ้นไป (แตกเป็นท่อน)  : {sum(r[1] >= 2 for r in big)}")