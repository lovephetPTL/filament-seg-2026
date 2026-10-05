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
