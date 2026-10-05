# Original Colab cells (archive, not runnable as-is)

These are the cells exactly as they were run in Google Colab while the solution was
developed. They contain Colab-only commands (`!pip`), absolute Drive paths and depend
on variables defined in other cells, so they are kept **for transparency only**.
The cleaned, runnable version of the same logic is in `src/filament/` and `notebooks/`.

| Original cell | Role | Now in |
|---|---|---|
| cell6a_final.py | tiling + date-grouped split | src/filament/data.py |
| cell6b_v2_normalized.py | dataset / normalisation | src/filament/dataset.py, dataset_utils.py |
| cell7_train_unet.py, cell7_continue_v2.py | training sessions | src/filament/train.py, scripts/train.py |
| cell12_overlap_inferenc.py, cell_setup_only.py, cell_gpu_full_eval.py | overlap inference | src/filament/tiling.py, infer.py |
| cell14_watershed.py, cell16_cache_and_tune.py | Watershed experiment (rejected) | not used in the final pipeline |
| cell19_pq_eval.py | our PQ implementation | src/filament/metrics.py |
| cell24_closing_ablation.py | closing ablation (A/B variants) | result in README; final step in postprocess.py |
| cell30_official_metric_crosscheck.py | check against organisers code | needs the organisers notebook |
| cell31_make_submission_closing31B.py | final submission | src/filament/pipeline.py |

Note: `cell12` has a default threshold of 0.30; the submitted predictions use 0.5
(as set in `cell_setup_only.py`, `cell31` and `src/filament/config.py`).
