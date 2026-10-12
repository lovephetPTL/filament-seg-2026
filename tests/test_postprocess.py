import numpy as np

from filament.postprocess import extract_instances


def _blank():
    return np.zeros((300, 300), np.uint8)


def test_small_pieces_are_removed():
    m = _blank()
    m[10:30, 10:30] = 1        # 400 px
    m[200:205, 200:205] = 1    # 25 px
    assert len(extract_instances(m, min_size=120)) == 1


def test_closing_groups_close_pieces_but_not_far_ones():
    m = _blank()
    m[100:120, 20:60] = 1      # piece A
    m[100:120, 70:110] = 1     # piece B, 10 px gap from A
    m[250:270, 250:290] = 1    # piece C, far away
    plain = extract_instances(m, 120, None)
    grouped = extract_instances(m, 120, closing_kernel=31)
    assert len(plain) == 3
    assert len(grouped) == 2


def test_grouping_never_adds_pixels():
    m = _blank()
    m[100:120, 20:60] = 1
    m[100:120, 70:110] = 1
    grouped = extract_instances(m, 120, closing_kernel=31)
    union = np.zeros_like(m, bool)
    for g in grouped:
        union |= g
    assert (union == m.astype(bool)).all()          # same pixels as the input mask
    assert grouped[0].sum() == m.sum()               # one instance made of both pieces


def test_viz_crop_and_overlay():
    from filament.viz import crop_box, overlay
    m = np.zeros((2048, 2048), bool)
    m[1990:2040, 1990:2040] = True                      # near the corner: window must be clipped to the image
    y0, x0 = crop_box(m)
    assert (y0, x0) == (2048 - 768, 2048 - 768)
    assert crop_box(np.zeros((2048, 2048), bool)) == (640, 640)
    gray = np.full((768, 768), 100, np.uint8)
    gt = np.zeros((768, 768), bool); gt[100:200, 100:200] = True
    pr = np.zeros((768, 768), bool); pr[300:400, 300:400] = True
    out = overlay(gray, [gt], [pr])
    assert out.shape == (768, 768, 3)
    assert tuple(out[100, 150]) == (0, 255, 255) and tuple(out[300, 350]) == (255, 0, 0)
