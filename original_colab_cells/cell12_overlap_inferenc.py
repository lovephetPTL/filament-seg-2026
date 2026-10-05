"""
Cell 12: Overlap Sliding-Window Inference (CPU Friendly, batched)
================================================================================
ตัดภาพเต็ม 2048x2048 เป็น tile แบบซ้อนทับกัน (stride 384, overlap 128px)
แล้วเอาผลทำนายของแต่ละ tile มา blend กันด้วย Gaussian weight (ขอบ tile มีน้ำหนักน้อยกว่ากลาง)
แก้จากต้นฉบับ: รวม 25 tile ต่อภาพเป็น batch เดียว ยิงเข้าโมเดลทีเดียว แทนที่จะทีละ tile
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

TILE_SIZE = 512
OVERLAP = 128            # overlap 128 พิกเซล (25%)
STRIDE = TILE_SIZE - OVERLAP   # 384 พิกเซล


def create_2d_gaussian_window(size, sigma=0.5):
    """สร้าง weight matrix นุ่มนวลสำหรับ blend ขอบภาพ (กลาง tile น้ำหนักสูง ขอบ tile น้ำหนักต่ำ)"""
    grid = np.linspace(-1, 1, size)
    x, y = np.meshgrid(grid, grid)
    d = np.sqrt(x * x + y * y)
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    return g


blend_weight = create_2d_gaussian_window(TILE_SIZE)

print(f"⚙️ ตั้งค่า Overlap Inference: Tile={TILE_SIZE}, Overlap={OVERLAP}, Stride={STRIDE}")


def predict_full_image_overlap(image_2048, model, device, threshold=0.30):
    """
    Inference ภาพใหญ่ 2048x2048 ด้วย Overlapping Window (Stride 384)
    พร้อม Gaussian soft blending บริเวณขอบทับซ้อน
    image_2048: tensor (C, H, W) ที่ normalize (ImageNet mean/std) แล้ว
    """
    H, W = image_2048.shape[-2:]
    prob_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    y_steps = list(range(0, H - TILE_SIZE + 1, STRIDE))
    if y_steps[-1] + TILE_SIZE < H:
        y_steps.append(H - TILE_SIZE)

    x_steps = list(range(0, W - TILE_SIZE + 1, STRIDE))
    if x_steps[-1] + TILE_SIZE < W:
        x_steps.append(W - TILE_SIZE)

    model.eval()
    # 🎯 รวมทุก tile ของภาพนี้เป็น batch เดียว ยิงเข้าโมเดลทีเดียว (เร็วกว่าทีละภาพมาก)
    coords = [(y, x) for y in y_steps for x in x_steps]
    tiles_batch = torch.stack(
        [image_2048[:, y:y + TILE_SIZE, x:x + TILE_SIZE] for y, x in coords]
    ).to(device)

    with torch.no_grad():
        tiles_prob = torch.sigmoid(model(tiles_batch)).cpu().squeeze(1).numpy()  # (25, H_tile, W_tile)

    for (y, x), tile_prob in zip(coords, tiles_prob):
        prob_map[y:y + TILE_SIZE, x:x + TILE_SIZE] += tile_prob * blend_weight
        weight_map[y:y + TILE_SIZE, x:x + TILE_SIZE] += blend_weight

    final_prob_map = prob_map / (weight_map + 1e-7)
    final_binary_mask = (final_prob_map > threshold).astype(np.float32)

    return final_prob_map, final_binary_mask
