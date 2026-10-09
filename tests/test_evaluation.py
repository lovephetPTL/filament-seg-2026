import numpy as np

from filament.evaluation import evaluate_pq
from filament.v3_data import photo_of

from synthetic_coco import make_coco


def _box(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_pq_pooled_over_annotator_images(tmp_path):
    ids = ["010101-20150101000000Lh", "010102-20150101000000Lh"]
    coco = make_coco(str(tmp_path), [(ids[0], [(100, 200, 100, 200)]),
                                     (ids[1], [(100, 200, 100, 200), (500, 600, 500, 600)])])
    perfect = {"20150101000000Lh": [_box(100, 200, 100, 200)]}
    r = evaluate_pq(coco, ids, perfect, photo_of)
    # annotator 1: TP 1; annotator 2: TP 1, FN 1 -> TP 2, FN 1, FP 0 -> PQ = 2 / (2 + 0.5)
    assert (r["TP"], r["FP"], r["FN"]) == (2, 0, 1) and abs(r["PQ"] - 0.8) < 1e-4
    # a useless extra prediction counts once per annotator-image: FP 2
    extra = {"20150101000000Lh": [_box(100, 200, 100, 200), _box(1000, 1100, 1000, 1100)]}
    r2 = evaluate_pq(coco, ids, extra, photo_of)
    assert (r2["TP"], r2["FP"], r2["FN"]) == (2, 2, 1) and r2["pred_pieces"] == 2


def test_no_prediction_gives_zero(tmp_path):
    ids = ["010101-20150101000000Lh"]
    coco = make_coco(str(tmp_path), [(ids[0], [(100, 200, 100, 200)])])
    r = evaluate_pq(coco, ids, {"20150101000000Lh": []}, photo_of)
    assert (r["TP"], r["FP"], r["FN"], r["PQ"]) == (0, 0, 1, 0.0)
