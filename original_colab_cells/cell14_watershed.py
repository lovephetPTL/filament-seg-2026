"""
Cell 14: Watershed Instance Separation
================================================================================
ปัญหาที่แก้: Semantic mask (จาก U-Net) บอกแค่ "พิกเซลนี้เป็น filament หรือไม่"
ถ้า filament 2 เส้นอยู่ใกล้กันจนโมเดลทายทับกันเป็นก้อนเดียว
Connected Components ธรรมดาจะนับเป็น 1 instance ทั้งที่จริงคือ 2 เส้น
Watershed ใช้ distance transform หา "จุดสันเขา" แยกก้อนที่เชื่อมกันออกจากกัน
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def label_naive(binary_mask):
    """วิธีเดิม: connected components ธรรมดา ไม่แยกก้อนที่ติดกัน"""
    labeled, n_instances = ndi.label(binary_mask)
    return labeled, n_instances


def label_watershed(binary_mask, min_distance=15):
    """
    แยก instance ด้วย Watershed:
    1. คำนวณ distance transform (ยิ่งอยู่กลางก้อนยิ่งค่าสูง)
    2. หา "จุดยอด" ของแต่ละก้อนย่อย (local maxima) เป็นเมล็ดเริ่มต้น (seed)
    3. ปล่อยน้ำท่วมจากแต่ละเมล็ด แบ่งอาณาเขตตามสันเขาของ distance transform
    """
    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32), 0

    distance = ndi.distance_transform_edt(binary_mask)

    # หาจุดสูงสุดในแต่ละก้อนย่อย ใช้เป็นเมล็ด (seed) สำหรับ watershed
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask.astype(bool))
    mask_seeds = np.zeros(distance.shape, dtype=bool)
    mask_seeds[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_seeds)

    labeled = watershed(-distance, markers, mask=binary_mask.astype(bool))
    n_instances = labeled.max()
    return labeled, n_instances


def count_gt_instances(img_id, coco):
    """นับจำนวน filament จริงจากเฉลย (จำนวน annotation ของภาพนี้)"""
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    return len(ann_ids)
