import numpy as np

from filament.tiling import blend_tiles, gaussian_window, tile_coords


def test_25_tiles_cover_whole_image():
    coords = tile_coords()
    assert len(coords) == 25
    cover = np.zeros((2048, 2048), bool)
    for y, x in coords:
        assert 0 <= y <= 2048 - 512 and 0 <= x <= 2048 - 512
        cover[y:y + 512, x:x + 512] = True
    assert cover.all()


def test_blending_a_constant_map_gives_the_constant_back():
    """Uniform probability 0.7 in every tile must come out as 0.7 everywhere."""
    coords = tile_coords()
    tiles = np.full((len(coords), 512, 512), 0.7, dtype=np.float32)
    out = blend_tiles(tiles, coords)
    assert np.allclose(out, 0.7, atol=1e-5)


def test_window_is_centre_heavy():
    w = gaussian_window(512)
    assert w[256, 256] > w[0, 0] and w.max() <= 1.0
