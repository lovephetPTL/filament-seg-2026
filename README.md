# Solar Filament Segmentation Challenge 2026 - U-Net (ResNet34) with overlap inference

Solution to the Kaggle *Solar Filament Segmentation Challenge 2026* (MAGFiLO, GONG H-alpha, 2048 x 2048 images).
Instance masks are obtained from a binary U-Net prediction, tiled and blended with overlapping windows,
then split into filaments with connected components.

| Submission | Description | Validation PQ (ours) | Public score |
|---|---|---|---|
| #1 baseline | overlap inference, threshold 0.5, components >= 120 px | 0.3825 | 0.31 |
| #2 Model A + closing | same + closing (k=31) used only to *group* pieces; submitted pixels unchanged | 0.3937 | 0.32 |
| #3 Model B + closing | baseline weights + 5 more epochs on the cleaned tile set (see "Supplementary experiments") | 0.3882 raw (repo pipeline with closing: not yet measured) | 0.33 |

**Final selection on Kaggle:** TODO (fill in after selecting the two files; a 0.01 difference on 180 test images is within noise).

Validation PQ was computed on our own date-grouped split (164 annotator-images = 103 photos). For the baseline it equals the
organisers Self-Evaluation code (0.3825); the closing variant was not cross-checked with it. The validation set was also used to choose
the checkpoint, the 120 px filter and the closing variant, so the validation numbers are optimistic. A gap between validation (0.38-0.39)
and the public score (0.31-0.32) remains unexplained.

## How to reproduce the submitted predictions
1. Get the test images from the Kaggle *Data* tab and place them, flat, in `test_images/` (180 files).
2. Get the weights from the **Releases** page of this repository (97.9 MB each) and put them in the repository root:
   `unet_resnet34_baseline_ep18.pth` (Model A) and `unet_resnet34_v3_control_ep5.pth` (Model B).
3. `pip install -r requirements.txt`
4. Run `notebooks/01_make_submission.ipynb`, or
   `python scripts/make_submission.py --images test_images --weights unet_resnet34_baseline_ep18.pth --out submission_modelA_closing31.csv`
   (Model B: `--weights unet_resnet34_v3_control_ep5.pth`)
   (`--no-closing` gives the baseline file).

No other file is needed. Training is **not** required to reproduce the predictions.

## Method
* **Data:** MAGFiLO training JSON (1,154 annotator-image entries = 707 unique photos, 8,199 filaments). 296 photos were annotated by 2-3 people with different polygons;
  every annotator-image is treated as its own sample.
* **Split:** grouped by capture date (`YYYYMMDD` of the file name), 85/15 on dates, seed 42 -> 990 train / 164 validation annotator-images (604 / 103 photos).
* **Tiles:** 512 x 512, no resizing (filaments are small). Train: all tiles with filament + 25 % of empty tiles; validation: all 2,624 tiles.
* **Input:** single-channel image replicated to 3 channels, scaled to [0,1], ImageNet mean/std.
* **Model:** `segmentation_models_pytorch.Unet`, ResNet34 encoder (ImageNet init), 1 output channel.
* **Loss:** 0.5 BCE + 0.5 Dice. **Optimiser:** AdamW, weight decay 1e-2, batch 8.
* **Inference:** 5 x 5 = 25 tiles per image (stride 384, 25 % overlap), Gaussian-weighted averaging of probabilities, threshold 0.5.
* **Instances:** 4-connected components; pieces smaller than 120 px are dropped (0.2 % of ground-truth filaments are that small).
  Final file: pieces are grouped through a closing with an elliptical 31 x 31 kernel, but only the original pixels are submitted.
* **Rejected:** Watershed splitting (default parameters over-segmented; after fixing a seed-suppression bug it scored below connected components),
  other thresholds (flat), 45 % overlap (no benefit).

### Closing ablation on validation (k fixed in advance: 7, 15, 31)
| Variant | PQ |
|---|---|
| baseline | 0.3825 |
| A: submit the closed mask (pixels added), k=7 / 15 / 31 | 0.3812 / 0.3828 / 0.3816 |
| B: closing only for grouping, k=7 / 15 / 31 | 0.3830 / 0.3877 / 0.3937 |

The gain of B (+0.011) comes mainly from fewer false positives (818 -> 719) with TP almost unchanged (774 -> 769). It is small and was selected on the validation set.

## Supplementary experiments and Model B
After the baseline we studied thin, feather-like filaments by hand and fine-tuned from the baseline weights. Details, tile lists and
result files are in `experiments/`; the training script is `scripts/train_v3.py`, the validation script `scripts/evaluate_val.py`.
Model B (`v3_control`, 5 epochs, lr 1e-4, random 90-degree rotation, 10 annotator-images and 173 tiles removed) reached validation PQ
0.3882 versus 0.3825 for Model A (single run per arm; differences below about 0.005 are treated as noise) and public 0.33 versus 0.32 with
closing. We do not claim that Model B is better. The feather-focused variants did not beat the baseline on validation (see `experiments/README.md`).

## Training history (honest account)
The checkpoint was produced in three consecutive Colab sessions. Each session resumed from the best checkpoint of the previous one and created a **new** optimiser.
The epoch counter in the logs (1-18) is therefore not the number of epochs the final weights have seen.

| Session | Log counter | Start | lr | Checkpoint chosen by | Outcome |
|---|---|---|---|---|---|
| 1 | 1-5 | ImageNet encoder | 3e-4 | validation loss | saved at epochs 1, 2, 4; best epoch 4 (val loss 0.1772, tile Dice 0.7568); epoch 5 had Dice 0.7758 but higher loss, so it was not saved |
| 2 | 6-10 in our notes (the screen showed 1-5) | session-1 checkpoint (val Dice 0.7568) | 1e-4 | validation loss | saved after epoch 1 (val loss 0.1715, Dice 0.7715) and epoch 2 (val loss 0.1646, Dice 0.7664); epochs 3-5 did not lower the loss (epoch 4 had a higher Dice, 0.7724, but was not saved). The checkpoint kept is epoch 2, Dice 0.7664 |
| 3 | 11-18 | session-2 checkpoint | 1e-4 | validation Dice | saved at 11, 12, 14, 17, 18; final tile Dice 0.7815 (re-measured on 2,624 tiles) |

The final weights have therefore received 4 + 2 + 8 = **14 epochs** of updates (session 1: best epoch 4; session 2: best epoch 2; session 3: all 8 epochs, the last one saved).

Tile Dice counts empty tiles as 1.0 and is therefore much higher than the filament-only Dice (~0.67) and PQ.

## Known limitations
* **Training is not exactly reproducible:** the random sampling of empty training tiles was not seeded in the original run (`scripts/train.py` seeds it now, but a re-cut gives a different tile set). The submitted checkpoint is provided instead.
* Hyper-parameters (epoch, 120 px filter, closing) were chosen with the validation set.
* Annotator disagreement is large; about 17 % of predicted pieces are supported by no annotator of multiply-annotated photos, 28 % are unsupported by a randomly chosen one.
* No augmentation, no architecture or loss search, input is a replicated single channel.

## Repository layout
```
src/filament/     config, tiling (blend), infer, postprocess, rle, pipeline, metrics, evaluation, data, v3_data, dataset, model, train
scripts/          make_submission.py, compare_submissions.py, evaluate_val.py, train.py, train_v3.py
experiments/      tile plan, result files and notes of the supplementary fine-tuning runs (Model B)
notebooks/        01_make_submission.ipynb  (the entry point for reviewers)
tests/            unit tests (pytest) for tiling/blending, post-processing, RLE, PQ and the split
original_colab_cells/   the experiment cells exactly as run in Colab (archive, see its README)
figures/          validation IoU / Dice / many-to-one plots of the baseline
```

## How to verify
* `pytest -q tests` checks the numpy-side logic (25-tile grid, blending of a constant map, closing never adds pixels, RLE round trip, hand-computed PQ cases, no date shared by train and validation).
* `scripts/compare_submissions.py` compares a regenerated CSV with the one uploaded to Kaggle.
* **End-to-end check (Colab T4, about 5 minutes):** `scripts/make_submission.py` regenerated the Model A file from the test images and the baseline weights (`best_baseline_unet.pth` = `unet_resnet34_baseline_ep18.pth`): 1,566 rows (same ids as the submitted file), 1,565 identical RLE strings, one differing mask with IoU 0.9983 against the submitted one (small numerical differences between runs), 8.7 filaments per image, 2 of 180 images without a prediction, 0 overlapping masks, 0 RLE round-trip failures.
* `infer.py`, `model.py`, `pipeline.py` were exercised by that run. `train.py`, `dataset.py` and `data.py` (training only) were refactored from the Colab cells and have NOT been re-run.

## Pinned versions
`requirements.txt` holds the versions printed by this snippet in the Colab session used for the check above:
```python
import importlib.metadata as m
for p in ["torch","segmentation-models-pytorch","numpy","scipy","opencv-python","opencv-python-headless","pycocotools","tqdm"]:
    try: print(f"{p}=={m.version(p)}")
    except Exception: print("#", p, "not installed")
```

## Data, license and citation
Data: MAGFiLO (Ahmadzadeh et al., 2024, Scientific Data 11:1031, doi:10.1038/s41597-024-03876-y), GONG data from NSO. Not included in this repository.
License: TODO - choose an OSI-approved license (e.g. MIT) before the competition closes (Kaggle rules require OSI-approved terms for shared code).
