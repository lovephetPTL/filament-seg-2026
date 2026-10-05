"""Overlapping sliding-window inference (25 tiles per image, Gaussian blending)."""
import numpy as np
import torch

from .config import IMG_SIZE, THRESHOLD
from .dataset_utils import normalize_np, read_rgb
from .tiling import blend_tiles, gaussian_window, tile_coords

_WINDOW = gaussian_window()


@torch.no_grad()
def predict_probability_map(image_chw, model, device, batch_size=25):
    """image_chw: float32 (3, 2048, 2048), already normalised -> (2048, 2048) probabilities."""
    model.eval()
    coords = tile_coords()
    tile = _WINDOW.shape[0]
    probs = []
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = torch.from_numpy(
            np.stack([image_chw[:, y:y + tile, x:x + tile] for y, x in chunk])
        ).to(device)
        probs.append(torch.sigmoid(model(batch)).cpu().squeeze(1).numpy())
    probs = np.concatenate(probs, axis=0)
    return blend_tiles(probs, coords, (IMG_SIZE, IMG_SIZE), _WINDOW)


def predict_binary_mask(image_path, model, device, threshold=THRESHOLD):
    img = normalize_np(read_rgb(image_path, expect_size=IMG_SIZE))
    prob = predict_probability_map(img, model, device)
    return prob, (prob > threshold).astype(np.uint8)
