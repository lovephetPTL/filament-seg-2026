"""Tile grid and Gaussian blending (pure numpy, no torch)."""
import numpy as np

from .config import IMG_SIZE, STRIDE, TILE_SIZE


def tile_coords(size=IMG_SIZE, tile=TILE_SIZE, stride=STRIDE):
    """Top-left corners (y, x) of the overlapping tiles. 2048 px -> 5 x 5 = 25 tiles."""
    steps = list(range(0, size - tile + 1, stride))
    if steps[-1] + tile < size:      # make sure the last tile reaches the border
        steps.append(size - tile)
    return [(y, x) for y in steps for x in steps]


def gaussian_window(size=TILE_SIZE, sigma=0.5):
    """2-D weight window: high in the tile centre, low at the tile edge."""
    grid = np.linspace(-1, 1, size)
    x, y = np.meshgrid(grid, grid)
    d = np.sqrt(x * x + y * y)
    return np.exp(-(d ** 2 / (2.0 * sigma ** 2)))


def blend_tiles(tile_probs, coords, shape=(IMG_SIZE, IMG_SIZE), window=None):
    """Weighted average of overlapping tile probabilities -> (H, W) float32 map."""
    tile = tile_probs[0].shape[-1]
    if window is None:
        window = gaussian_window(tile)
    prob_map = np.zeros(shape, dtype=np.float32)
    weight_map = np.zeros(shape, dtype=np.float32)
    for (y, x), p in zip(coords, tile_probs):
        prob_map[y:y + tile, x:x + tile] += p * window
        weight_map[y:y + tile, x:x + tile] += window
    return prob_map / (weight_map + 1e-7)
