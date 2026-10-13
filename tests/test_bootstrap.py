import numpy as np

from filament.bootstrap import paired_bootstrap, pq_from_sums
from filament.evaluation import evaluate_pq, per_photo_counts
from filament.v3_data import photo_of

from synthetic_coco import make_coco


def test_pq_from_sums():
    assert abs(pq_from_sums([2, 1.6, 1, 1]) - 1.6 / 3.0) < 1e-9


def test_identical_models_give_zero_difference():
    rng = np.random.default_rng(1)
    c = np.stack([rng.integers(1, 10, 40), rng.uniform(0.5, 1, 40) * 5,
                  rng.integers(0, 5, 40), rng.integers(0, 5, 40)], 1).astype(float)
    r = paired_bootstrap(c, c.copy(), n_boot=300, seed=0)
    assert r["diff_B_minus_A"] == 0 and r["ci95_diff"] == [0.0, 0.0] and r["share_of_resamples_B_better"] == 0.0


def test_clearly_better_model_has_interval_above_zero():
    n = 60
    base = np.tile([5.0, 3.5, 3.0, 3.0], (n, 1))
    better = np.tile([8.0, 6.0, 1.0, 1.0], (n, 1))
    r = paired_bootstrap(base, better, n_boot=300, seed=0)
    assert r["ci95_diff"][0] > 0 and r["share_of_resamples_B_better"] == 1.0


def test_per_photo_counts_sum_to_evaluate_pq(tmp_path):
    ids = ["010101-20150101000000Lh", "010102-20150101000000Lh", "020101-20150202000000Lh"]
    coco = make_coco(str(tmp_path), [(ids[0], [(100, 200, 100, 200)]),
                                     (ids[1], [(100, 200, 100, 200), (500, 600, 500, 600)]),
                                     (ids[2], [(300, 400, 300, 400)])])

    def box(y0, y1, x0, x1):
        m = np.zeros((2048, 2048), bool)
        m[y0:y1, x0:x1] = True
        return m
    inst = {"20150101000000Lh": [box(100, 200, 100, 200), box(900, 950, 900, 950)],
            "20150202000000Lh": [box(300, 400, 300, 380)]}
    counts = per_photo_counts(coco, ids, inst, photo_of)
    total = sum(counts.values())
    ref = evaluate_pq(coco, ids, inst, photo_of)
    assert (int(total[0]), int(total[2]), int(total[3])) == (ref["TP"], ref["FP"], ref["FN"])
    assert abs(pq_from_sums(total) - ref["PQ"]) < 1e-4
