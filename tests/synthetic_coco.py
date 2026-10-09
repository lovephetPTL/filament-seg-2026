"""Tiny synthetic COCO dataset for tests (2048 x 2048 images with rectangular filaments)."""
import os

import cv2
import numpy as np
from pycocotools.coco import COCO


def rect_poly(y0, y1, x0, x1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def make_coco(tmp_path, spec):
    """spec: list of (annotator_image_id, [rects (y0, y1, x0, x1)]). Writes grey images and returns COCO."""
    images, anns, aid = [], [], 1
    for i, (iid, rects) in enumerate(spec):
        photo = iid.split("-", 1)[1]
        fn = photo + ".jpeg"
        path = os.path.join(tmp_path, fn)
        if not os.path.exists(path):
            cv2.imwrite(path, np.full((2048, 2048), 100, np.uint8))
        images.append({"id": iid, "file_name": fn, "height": 2048, "width": 2048})
        for r in rects:
            y0, y1, x0, x1 = r
            anns.append({"id": aid, "image_id": iid, "category_id": 1, "iscrowd": 0,
                         "segmentation": rect_poly(*r), "area": float((y1 - y0) * (x1 - x0)),
                         "bbox": [x0, y0, x1 - x0, y1 - y0]})
            aid += 1
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "f"}]}
    coco.createIndex()
    return coco
