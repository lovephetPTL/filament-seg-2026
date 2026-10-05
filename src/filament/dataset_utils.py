"""Image loading shared by training and inference."""
import cv2
import numpy as np

from .config import IMAGENET_MEAN, IMAGENET_STD


def read_rgb(path, expect_size=None):
    """Read an image with OpenCV as RGB uint8 (H, W, 3).

    The GONG images are single-channel; OpenCV replicates the channel three
    times, which is what the 3-channel pretrained ResNet34 encoder receives.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if expect_size is not None and img.shape[:2] != (expect_size, expect_size):
        raise ValueError(f"{path}: expected {expect_size}x{expect_size}, got {img.shape[:2]}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize_np(img_rgb_uint8):
    """uint8 (H, W, 3) -> float32 (3, H, W), scaled to [0,1] and ImageNet-normalised."""
    x = img_rgb_uint8.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    return (x - mean) / std
