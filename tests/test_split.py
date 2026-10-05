from filament.data import split_by_date


def _images():
    out, i = [], 0
    for d in range(40):                       # 40 dates, 1-3 annotator-images each
        for a in range(1 + d % 3):
            out.append({"id": f"0{a}0101-2015{d // 4 + 1:02d}{d % 4 + 10:02d}063134Lh",
                        "file_name": f"2015{d // 4 + 1:02d}{d % 4 + 10:02d}063134Lh.jpeg"})
            i += 1
    return out


def test_no_date_in_both_sets_and_deterministic():
    imgs = _images()
    tr, va = split_by_date(imgs)
    assert len(tr) + len(va) == len(imgs) and set(tr).isdisjoint(va)
    date = {im["id"]: im["file_name"][:8] for im in imgs}
    assert {date[i] for i in tr}.isdisjoint({date[i] for i in va})
    assert split_by_date(imgs) == (tr, va)     # same seed -> same split
