# cell31_make_submission_closing31B.py
import os, csv, zipfile, shutil, cv2, numpy as np, torch
from scipy import ndimage as ndi
from pycocotools import mask as mask_util
from tqdm import tqdm

D = "/content/drive/MyDrive/Colab Filament"
TEST_DIR, OUT = "test_images", "submission_closing31B.csv"
MIN_SIZE, THR, K = 120, 0.5, 31

# แตกไฟล์ test แบบแบน (ข้ามถ้ามีแล้ว)
if not os.path.isdir(TEST_DIR) or len(os.listdir(TEST_DIR)) < 100:
    os.makedirs(TEST_DIR, exist_ok=True)
    with zipfile.ZipFile(os.path.join(D, "test_images.zip")) as z:
        for m in z.namelist():
            if m.endswith("/") or not m.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            with z.open(m) as s, open(os.path.join(TEST_DIR, os.path.basename(m)), "wb") as d:
                d.write(s.read())
files = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpeg", ".jpg", ".png")))
print("จำนวนภาพ test:", len(files), "(ควรเป็น 180)")

kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
enc = lambda m: mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("utf-8")
dec = lambda s: mask_util.decode({"size": [2048, 2048], "counts": s.encode("utf-8")}).astype(bool)

def load_t(p):
    img = cv2.imread(p)
    assert img is not None and img.shape[:2] == (2048, 2048), p
    t = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return (t - IMAGENET_MEAN) / IMAGENET_STD

rows, per_img, empty, overlap_bad, decode_bad = [], {}, [], 0, 0
for fn in tqdm(files):
    stem = os.path.splitext(fn)[0]
    _, binary = predict_full_image_overlap(load_t(os.path.join(TEST_DIR, fn)), model, device, THR)
    b = binary.astype(np.uint8)
    lab, n = ndi.label(cv2.morphologyEx(b, cv2.MORPH_CLOSE, kern))
    lab = lab * b                                   # เก็บเฉพาะพิกเซลเดิม (ไม่ขยายขอบ)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    tot = np.zeros((2048, 2048), np.uint8)
    k = 0
    for i in range(1, n + 1):
        if sizes[i] >= MIN_SIZE:
            m = lab == i
            k += 1
            s = enc(m)
            decode_bad += int(not (dec(s) == m).all())
            tot += m
            rows.append((f"{stem}_{k}", s))
    overlap_bad += int(tot.max() > 1)
    per_img[stem] = k
    if k == 0:
        empty.append(stem)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["filament_id", "segmentation_rle"]); w.writerows(rows)
shutil.copy(OUT, os.path.join(D, OUT))

ids = [r[0] for r in rows]; stems = {os.path.splitext(f)[0] for f in files}
c = np.array(list(per_img.values()))
print(f"แถว: {len(rows)} | id ไม่ซ้ำ: {len(set(ids)) == len(ids)} | id ตรงชื่อภาพ: {all(i.rsplit('_', 1)[0] in stems for i in ids)}")
print(f"ไม่มีเครื่องหมายคำพูด: {all(chr(34) not in s and chr(39) not in s for _, s in rows)}")
print(f"ภาพที่ mask ทับกัน: {overlap_bad} (ต้องเป็น 0) | แถวที่ถอดรหัสกลับไม่ตรง: {decode_bad} (ต้องเป็น 0)")
print(f"เส้นต่อภาพ: เฉลี่ย {c.mean():.2f} | มัธยฐาน {np.median(c):.0f} | สูงสุด {c.max()} | (ไฟล์ baseline เคยได้ 1690 แถว เฉลี่ย 9.39)")
print(f"ภาพที่ไม่มีแถวเลย: {len(empty)}")