# cell30_official_metric_crosscheck.py
# ให้ "โค้ดของผู้จัด" คิด PQ จากภาพ val ของเรา เพื่อเทียบกับตัวคิดของเรา
# แล้วคำนวณช่วงแกว่งแบบ pooled (ถ่วงตามผู้ขีด) เมื่อสุ่ม 90 ภาพ และเซฟกราฟ 3 แบบสำหรับรายงาน
# ต้องมี: coco, val_ids (จาก mini setup), โค้ดผู้จัดที่รันแล้ว, Drive ต่ออยู่ (mask_cache_val.npz)
# ไม่ใช้ GPU  เวลา: หลายนาที (ยังไม่ได้วัด)

import os, cv2, numpy as np, pandas as pd
from collections import defaultdict
from scipy import ndimage as ndi
from pycocotools import mask as mask_util
from tqdm import tqdm
import matplotlib.pyplot as plt

for need in ("get_overlap_df", "get_pq_score", "plot_distribution", "plot_m2n_counts",
             "fp_count_hit", "fn_count_hit"):
    assert need in globals(), f"ยังไม่มี {need}: วางและรันโค้ดผู้จัดก่อนเซลล์นี้"

DRIVE_DIR = "/content/drive/MyDrive/Colab Filament"
METHOD = "baseline"    # "baseline" = Naive + size120 (ตรงกับไฟล์ที่ส่ง) | "closing31B" = ทดลอง
MIN_SIZE = 120
data = np.load(os.path.join(DRIVE_DIR, "mask_cache_val.npz"))


def enc(m):
    return mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("utf-8")


def instances(binary):
    b = binary.astype(np.uint8)
    if METHOD == "closing31B":
        closed = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        lab, n = ndi.label(closed)
        lab = lab * b
    else:
        lab, n = ndi.label(b)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    return [lab == i for i in range(1, n + 1) if sizes[i] >= MIN_SIZE]


# ---- สร้าง gt_df / pred_df รูปแบบเดียวกับ submission ----
by_file = defaultdict(list)
for i in val_ids:
    by_file[coco.loadImgs([i])[0]["file_name"]].append(i)

gt_rows, pred_rows = [], []
for i in val_ids:
    k = 0
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[i])):
        m = coco.annToMask(ann)
        if m.sum() == 0:
            continue
        k += 1
        gt_rows.append((f"{i}_{k}", enc(m)))          # เช่น 010401-20141106195854Bh_1
for fname, ids in tqdm(by_file.items(), desc="สร้าง pred_df"):
    stem = os.path.splitext(fname)[0]
    for k, m in enumerate(instances(data[ids[0]]), start=1):
        pred_rows.append((f"{stem}_{k}", enc(m)))     # เช่น 20141106195854Bh_1

gt_df = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])
pred_df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])
print(f"gt_df: {len(gt_df)} แถว (ควรเป็น 1155) | pred_df: {len(pred_df)} แถว (baseline ควรเป็น 1035)")

# ---- ให้โค้ดผู้จัดคิด ----
overlap_df = get_overlap_df(gt_df, pred_df)
pq_off = get_pq_score(overlap_df)
print(f"\nPQ จากโค้ดผู้จัด: {pq_off:.4f}   (baseline ของเราคือ 0.3825, closing31B คือ 0.3937)")

# ---- ช่วงแกว่งแบบ pooled: สุ่มภาพ 90 ใบ โดยเอาผู้ขีดทุกคนของภาพนั้นไปด้วย ----
rows = []
for r in overlap_df.itertuples(index=False):
    img = r.annotator_image.split("-", 1)[1]
    if r.n_pred == 0:
        rows.append((img, 0, 0.0, 0, r.n_gt))
        continue
    hit = r.iou_matrix > 0.5
    rows.append((img, int(hit.sum()), float(r.iou_matrix[hit].sum()),
                 fp_count_hit(hit), fn_count_hit(hit)))
agg = pd.DataFrame(rows, columns=["img", "tp", "iou", "fp", "fn"]).groupby("img").sum()
A = agg[["tp", "iou", "fp", "fn"]].to_numpy(dtype=float)
pq_sum = lambda s: s[1] / (s[0] + 0.5 * s[2] + 0.5 * s[3])
print(f"เช็ค PQ รวมจากตารางนี้: {pq_sum(A.sum(0)):.4f} (ต้องเท่ากับบรรทัดบน)")
rng = np.random.default_rng(0)
vals = [pq_sum(A[rng.integers(0, len(A), 90)].sum(0)) for _ in range(5000)]
lo, mid, hi = np.percentile(vals, [2.5, 50, 97.5])
print(f"PQ ถ้าสุ่ม 90 ภาพจาก val: กลาง {mid:.3f} | ช่วง 95% [{lo:.3f}, {hi:.3f}]  (test สาธารณะได้ 0.31)")

# ---- กราฟ 3 แบบตามรูบริก (เซฟลง Drive) ----
for name, fn in (("iou", lambda: plot_distribution(overlap_df["iou_matrix"], "IoU Distribution", "IoU")),
                 ("dice", lambda: plot_distribution(overlap_df["dice_matrix"], "Dice Distribution", "Dice")),
                 ("m2n", lambda: plot_m2n_counts(overlap_df, "Many to One and One to Many Distribution"))):
    plt.figure(figsize=(7, 4.5))
    fn()
    plt.tight_layout()
    plt.savefig(os.path.join(DRIVE_DIR, f"val_{METHOD}_{name}.png"), dpi=110)
    plt.show()
print("เซฟกราฟลง Drive แล้ว")