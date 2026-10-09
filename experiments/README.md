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

Sources (all in `results/`): `val_results_organizer_code.json` (baseline, feather_x4), `val_results_finetune.json` (ft_union, ft_control),
`val_results_v3.json` (v3_main, v3_control). Public scores are shown by Kaggle with two decimals on part of the test set.

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
