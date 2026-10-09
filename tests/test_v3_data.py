from filament.v3_data import build_training_tiles, load_plan, photo_of

from synthetic_coco import make_coco

IDS = ["010101-20150101000000Lh",   # k=0  two annotators of the same photo
       "010102-20150101000000Lh",   # k=1
       "020101-20150202000000Lh",   # k=2  dropped image
       "030101-20150303000000Lh"]   # k=3  no filament at all


def _coco(tmp_path):
    return make_coco(str(tmp_path), [
        (IDS[0], [(600, 700, 600, 700)]),                  # tile j=5 (row 1, col 1)
        (IDS[1], [(620, 720, 620, 720)]),                  # slightly different outline, same photo
        (IDS[2], [(10, 50, 10, 50)]),
        (IDS[3], []),
    ])


def _plan(**kw):
    p = {"feather": {20000 + 5}, "exclude": set(), "drop_images": {IDS[2]}}
    p.update(kw)
    return p


def test_photo_of():
    assert photo_of(IDS[0]) == "20150101000000Lh"


def test_dropped_image_is_skipped_and_feather_is_repeated(tmp_path):
    coco = _coco(tmp_path)
    imgs, masks, items, st = build_training_tiles(coco, IDS, _plan(), str(tmp_path), feather_repeat=3)
    repeated = [i for i in set(items) if items.count(i) == 3]
    assert st["feather"] == 1 and len(repeated) == 1            # the feather tile appears 3 x per epoch
    assert all(m.shape == (512, 512) for m in masks)
    assert st["filament"] == 1                                  # tile j=5 of the 2nd annotator
    # image 3 (k=2) is dropped: its filament at (10..50) must not appear in any tile
    assert not any(m[10:50, 10:50].all() and m.sum() == 40 * 40 for m in masks)


def test_excluded_tile_is_not_used(tmp_path):
    coco = _coco(tmp_path)
    plan = _plan(exclude={20000 + 16 + 5})                      # tile 5 of image k=1
    _, _, items, st = build_training_tiles(coco, IDS, plan, str(tmp_path))
    assert st["excluded"] == 1 and st["filament"] == 0


def test_empty_tile_sampling_is_seeded(tmp_path):
    coco = _coco(tmp_path)
    a = build_training_tiles(coco, IDS, _plan(), str(tmp_path), seed=42)
    b = build_training_tiles(coco, IDS, _plan(), str(tmp_path), seed=42)
    assert a[3] == b[3] and a[2] == b[2]
    kept = a[3]["empty_kept"]
    assert 0 <= kept < 45                                       # 15+15+15 empty candidates, ~25 % kept


def test_blue_wins_changes_the_feather_label(tmp_path):
    coco = _coco(tmp_path)
    own = build_training_tiles(coco, IDS, _plan(), str(tmp_path), feather_repeat=2, blue_wins=False)
    blue = build_training_tiles(coco, IDS, _plan(), str(tmp_path), feather_repeat=2, blue_wins=True)
    assert own[3]["feather_label_changed"] == 0
    assert blue[3]["feather_label_changed"] == 1               # owner outline replaced by the other annotator one
    f_own = [i for i in set(own[2]) if own[2].count(i) == 2][0]
    f_blue = [i for i in set(blue[2]) if blue[2].count(i) == 2][0]
    assert own[1][f_own].sum() == 100 * 100                    # owner label: 100 x 100 box
    assert blue[1][f_blue].sum() == 100 * 100 and (own[1][f_own] != blue[1][f_blue]).any()
