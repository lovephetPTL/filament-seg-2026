"""Compare two submission CSVs (e.g. the file uploaded to Kaggle vs. the regenerated one).

  python scripts/compare_submissions.py submitted.csv regenerated.csv
Reports row counts, id overlap and the share of identical RLE strings, then the
pixel-level agreement of the masks that differ.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402

from filament.rle import decode_rle  # noqa: E402


def read(path):
    with open(path, newline="") as f:
        return {r["filament_id"]: r["segmentation_rle"] for r in csv.DictReader(f)}


def main(a, b):
    A, B = read(a), read(b)
    common = set(A) & set(B)
    same = sum(A[k] == B[k] for k in common)
    print(f"rows: {len(A)} vs {len(B)} | ids only in first: {len(set(A) - set(B))} | "
          f"only in second: {len(set(B) - set(A))}")
    print(f"identical RLE strings: {same}/{len(common)}")
    diffs = [k for k in sorted(common) if A[k] != B[k]]
    if diffs:
        ious = []
        for k in diffs:
            x, y = decode_rle(A[k]), decode_rle(B[k])
            ious.append((x & y).sum() / max((x | y).sum(), 1))
        print(f"differing masks: {len(diffs)} | mean IoU between them: {np.mean(ious):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
