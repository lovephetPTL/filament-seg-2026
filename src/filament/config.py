"""All fixed settings of the submitted pipeline in one place.

Values are exactly those used for the submitted predictions
(see README, section "What was submitted").
"""

# ---- image / tiling -------------------------------------------------------
IMG_SIZE = 2048          # all GONG images are 2048 x 2048 pixels
TILE_SIZE = 512
OVERLAP = 128            # 25 % overlap between neighbouring tiles
STRIDE = TILE_SIZE - OVERLAP   # 384 px -> 5 x 5 = 25 tiles per image

# ---- inference / post-processing -----------------------------------------
THRESHOLD = 0.5          # sigmoid threshold for the binary mask
MIN_SIZE = 120           # predicted pieces smaller than this (pixels) are dropped
CLOSING_KERNEL = 31      # elliptical kernel used ONLY to group nearby pieces

# ---- input normalisation (ImageNet statistics, matches the pretrained encoder)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- data preparation (training only) ------------------------------------
SPLIT_SEED = 42          # seed of the date-grouped train/validation split
VAL_RATIO = 0.15         # share of *dates* that go to validation
EMPTY_KEEP_RATIO = 0.25  # share of empty training tiles that are kept
