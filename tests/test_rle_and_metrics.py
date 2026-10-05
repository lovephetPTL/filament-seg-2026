import numpy as np

from filament.metrics import match_instances, pq_from_counts
from filament.rle import decode_rle, encode_mask


def test_rle_roundtrip():
    m = np.zeros((2048, 2048), bool)
    m[100:400, 50:90] = True
    m[1500:1510, 1500:1900] = True
    s = encode_mask(m)
    assert chr(34) not in s and chr(39) not in s
    assert (decode_rle(s) == m).all()


def _gt(box, shape=(100, 100)):
    g = np.zeros(shape, np.uint8)
    y0, y1, x0, x1 = box
    g[y0:y1, x0:x1] = 1
    return np.flatnonzero(g)


def test_pq_hand_computed_cases():
    # one GT (10x10 = 100 px), one prediction covering 80 of them -> IoU = 0.8
    pred = np.zeros((100, 100), np.int32)
    pred[0:10, 0:8] = 1
    tp, s, npred, ngt = match_instances(pred, [_gt((0, 10, 0, 10))])
    assert (tp, npred, ngt) == (1, 1, 1) and abs(s - 0.8) < 1e-9
    assert abs(pq_from_counts(tp, s, npred, ngt) - 0.8) < 1e-9

    # add a second, unmatched prediction: PQ = 0.8 / (1 + 0.5) = 0.5333
    pred[50:60, 50:60] = 2
    tp, s, npred, ngt = match_instances(pred, [_gt((0, 10, 0, 10))])
    assert abs(pq_from_counts(tp, s, npred, ngt) - 0.8 / 1.5) < 1e-9

    # a GT with no prediction at all: PQ = 0.8 / (1 + 0.5) as well (one FN instead of one FP)
    tp, s, npred, ngt = match_instances(np.where(pred == 1, 1, 0).astype(np.int32),
                                        [_gt((0, 10, 0, 10)), _gt((50, 60, 50, 60))])
    assert abs(pq_from_counts(tp, s, npred, ngt) - 0.8 / 1.5) < 1e-9


def test_iou_must_exceed_half():
    pred = np.zeros((100, 100), np.int32)
    pred[0:10, 0:5] = 1                      # IoU exactly 0.5 -> not a match
    tp, *_ = match_instances(pred, [_gt((0, 10, 0, 10))])
    assert tp == 0
