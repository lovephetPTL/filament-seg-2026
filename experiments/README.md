# Supplementary experiments (fine-tuning from the baseline)

Goal: help the model with thin, feather-like filaments that the baseline often leaves out. All runs start from the baseline weights
(Model A, 14 epochs of updates), use AdamW (weight decay 1e-2), batch 8, loss 0.5 BCE + 0.5 Dice, seed 42, a Colab T4 GPU.
**Rule fixed before seeing any result:** judge only the LAST epoch of each run; a PQ difference below about 0.005 counts as noise.
Every number is from ONE training run per arm (no repeated seeds, no paired confidence interval).

## Training tiles
`tile_plan.json` holds the lists used by `src/filament/v3_data.py` (tile number = 20000 + 16 x index in the frozen order of the 990
training annotator-images + 4 x row + column):
* `dropped_images` (10): annotator-images whose ground truth was marked but where nothing is visible (code B). Removed completely.
* `holdout_ACD_tiles` (124) and `removed_Bstar_tiles` (112): 173 tiles in total (63 are in both lists) that are left out of training.
  The A/C/D tiles are kept as a held-out set for visual checks; B* tiles are cuts of a filament by the tile grid.
* `feather_tiles` (448): thin, feather-like filaments chosen by eye (one person, not blind), used with a repeat factor.
* `stats`: counts printed by the first feather run (`exp_feather_x4`, blue-wins labels, repeat 4).
Training set for Model B: 5,067 filament tiles + 2,479 empty tiles (seeded 25 %) + 448 feather tiles = **7,994 samples per epoch**.

## Runs and validation PQ (organisers metric, 164 annotator-images, 1,155 GT filaments)
| Run | Recipe | PQ | TP | FP | FN | Public (closing k=31) |
|---|---|---|---|---|---|---|
| Model A, baseline | 14 epochs, see main README | 0.3825 | 774 | 818 | 381 | 0.32 (raw file: 0.31) |
| feather_x4, epoch 7 | big set, feather x4, blue-wins labels, no rotation, lr 1e-4, 7 epochs | 0.3896 | 752 | 698 | 403 | 0.32 |
| ft_union, epoch 5 | fine-tune on 448 feather + 150 empty tiles only, blue-wins, rotation, BatchNorm frozen, lr 1e-5, 5 epochs | 0.2791 | 766 | 1647 | 389 | not submitted |
| ft_control, epoch 5 | same, owner labels | 0.2826 | 835 | 1939 | 320 | not submitted |
| v3_main, epoch 5 | big set, feather x4, blue-wins, rotation, lr 1e-4, 5 epochs | 0.3640 | 816 | 1068 | 339 | 0.31 |
| **Model B = v3_control, epoch 5** | big set, feather x1, owner labels, rotation, lr 1e-4, 5 epochs | 0.3882 | 765 | 756 | 390 | 0.33 |

Sources (all in `results/`; the paired bootstrap result is described further below): `val_results_organizer_code.json` (baseline, feather_x4), `val_results_finetune.json` (ft_union, ft_control),
`val_results_v3.json` (v3_main, v3_control). Public scores are shown by Kaggle with two decimals on part of the test set.

## Validation PQ of the final pipeline (closing k=31) and an exploratory probability filter
File: `results/val_results_meanprob.json` (computed in Colab by the authors own evaluation cell with the organisers PQ code; the Model A closing
numbers equal the earlier 0.3937 measurement, TP 769 / FP 719 / FN 386, which supports that the evaluation is the same as the repository pipeline).
`mean_prob >= 0.85` means: a predicted piece is kept only if the mean sigmoid probability of its pixels is at least 0.85. One value, fixed in advance
(no sweep). The filter code is **not in the repository yet** and is NOT part of the submitted files.

| Model | Post-processing | PQ | TP | FP | FN | Pieces |
|---|---|---|---|---|---|---|
| A (baseline) | closing k=31 | 0.3937 | 769 | 719 | 386 | 967 |
| A (baseline) | closing k=31 + mean_prob >= 0.85 | 0.4033 | 763 | 645 | 392 | 913 |
| B (v3_control) | closing k=31 | 0.3983 | 763 | 676 | 392 | 936 |
| B (v3_control) | closing k=31 + mean_prob >= 0.85 | 0.4052 | 753 | 608 | 402 | 886 |

Reading: the filter removes about 70 false positives per model and loses 6-10 true positives (PQ +0.0096 for A, +0.0069 for B). Both are measured on the
same validation set that was used for earlier choices, one run per model, no paired interval: treat them as promising, not proven. With closing, Model B
is +0.0046 above Model A on validation (+0.0019 with the filter), which is within the noise level we set (about 0.005); the public scores (0.33 vs 0.32) agree
in sign but are also within noise. The filter has not been applied to the test images or submitted to Kaggle.

## Later experiments (v4 soft target, hysteresis threshold): negative results
Source: the authors Colab runs (Project Log Part 5); the result files and the code for these runs are **not in this repository yet**.
Bar fixed in advance for every test: PQ must beat Model B with closing (0.3983) by at least 0.01, i.e. reach 0.4083. No test reached it.
* **v4 soft target:** same recipe as Model B, only the label changes (mean of the masks of all annotators of the same photo, 599 training photos,
  42 % of them with 2-3 annotators). PQ 0.3939 (TP 780, FP 733, FN 375, 990 pieces): more TP and fewer FN than Model B but more FP; -0.0044 vs Model B, within noise.
  Remark: the Dice loss of segmentation_models_pytorch 0.5.0 gave values on soft labels that matched neither the textbook soft Dice nor a binarised target
  (cause not investigated), so v4 used a hand-written soft Dice that equals smp on 0/1 labels.
* **Hysteresis threshold** (keep a region of prob > t_lo only if it contains a pixel > t_hi), final pipeline otherwise unchanged:

| t_lo, t_hi | Model B PQ | v4 PQ |
|---|---|---|
| 0.5, 0.5 (code check, equals the old pipeline) | 0.3983 | 0.3939 |
| 0.4, 0.6 | 0.4021 | 0.3953 |
| 0.4, 0.7 | 0.4019 | 0.3962 |
| 0.5, 0.7 | 0.3986 | 0.3956 |
| 0.2, 0.4 | 0.3941 | 0.3912 |

  Best gain about +0.004 (within noise); lowering t_lo to 0.2 made both models worse. Pairs (0.4, 0.6) for Model B and (0.3, 0.6) for v4 were also uploaded to
  Kaggle as exploratory files (public 0.33 and 0.32); the (0.3, 0.6) pair was chosen after seeing validation results and was not measured on validation.
* **Visual check of the pieces removed by `mean_prob >= 0.85` (Model B, 50 pieces, one observer, not blind):** 46 labelled real filament, 1 not a filament, 3 unsure;
  only 8 of the 50 match a ground-truth filament. This suggests the filter would also remove faint real structures. It is an observation, not a measurement.

## Paired bootstrap, Model B versus Model A (`results/paired_bootstrap.json`)
Final pipeline (closing k=31), 103 validation photos, 10,000 resamples, seed 0, all annotator-images of a photo drawn together.
PQ(A) 0.3937 (95 % interval [0.3712, 0.4147]), PQ(B) 0.3983 ([0.3780, 0.4181]); difference B - A = +0.0046, 95 % interval [-0.0086, +0.0175],
B better in 76.6 % of the resamples. Rule fixed before the result: the difference counts as supported only if the interval excludes 0; it does not.
The intervals describe the sampling of validation photos only; one training run per model and validation-based model selection are not covered.
The width of the interval (about +-0.013) is larger than the 0.005 working threshold used in the experiments above, so those differences are within noise as well.

## What the results say (and do not say)
* No fine-tuned model is better than the baseline beyond noise. Model B is +0.0057 PQ on validation, just above our 0.005 threshold, one run per arm.
* v3_main versus v3_control differ only in feather repetition and blue-wins labels: TP +51, FN -51, FP +312, PQ -0.024. The model finds more
  filaments it used to miss entirely but draws many more pieces that no annotator marked. Whether those pieces are real filaments is **not known**
  (a visual check of all of them was not done).
* Fine-tuning on the feather tiles alone (ft_union / ft_control) ruins precision (FP about 1,650-1,940). Judgement: the training set was almost
  only filament tiles; this was not tested separately.
* Validation scores each annotator-image against its owner only, so a model trained to follow other annotators outlines can look worse on it.

## Reproduce Model B
```
python scripts/train_v3.py --json MAGFiLO_1.0_Annotations_kaggle2026_train.json --images train \
       --start-weights unet_resnet34_baseline_ep18.pth --run-name exp_v3_control
python scripts/evaluate_val.py --json MAGFiLO_1.0_Annotations_kaggle2026_train.json --images train \
       --weights runs/exp_v3_control/ep5.pth --closing 31
```
`train_v3.py` and `evaluate_val.py` are cleaned ports of the Colab cells and have not been re-run. The shipped weights are the reference.
