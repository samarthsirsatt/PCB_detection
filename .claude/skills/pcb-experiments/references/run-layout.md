# Ultralytics run directory anatomy

```
runs/<name>/
  results.csv              per-epoch row: epoch, train/*_loss, metrics/precision(B),
                            metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), lr/*
  args.yaml                every argument .train() was actually called with — the ground
                            truth for what hyperparameters produced this run
  weights/best.pt           checkpoint at the best-scoring epoch (by the configured metric,
                            usually fitness = weighted mAP50 + mAP50-95)
  weights/last.pt           final epoch's checkpoint — needed to resume a requeued job
                            (see prajna-jobs/templates/resumable_job.sbatch)
  confusion_matrix.png
  confusion_matrix_normalized.png
  PR_curve.png, F1_curve.png, P_curve.png, R_curve.png
  labels.jpg, labels_correlogram.jpg
  train_batch*.jpg, val_batch*_labels.jpg, val_batch*_pred.jpg
```

## What matters for a writeup vs. what's disposable

**Keep (small, needed for reporting or resuming):**
- `results.csv`, `args.yaml` — the actual numbers and how they were produced.
- `weights/best.pt` — for any later inference/comparison.
- `confusion_matrix*.png`, `PR_curve.png` — the plots a writeup actually cites.

**Safe to discard once metrics are captured (large, purely diagnostic):**
- `weights/last.pt` — only needed if you might resume this exact run; delete once you're
  confident the run is finished for good.
- `train_batch*.jpg`, `val_batch*.jpg` — training-time visualization dumps, useful for
  eyeballing during a run, not needed after.

Given the output placement rule in `PRAJNA_CONFIG.md` §8 (`runs/` lives on `/home`, not
`/scratch`), trimming `last.pt` and the batch visualization JPEGs periodically keeps quota
pressure down without touching anything the reaper wouldn't have deleted anyway.

## Cross-validation folds

A 5-fold CV run produces 5 directories, typically named `cv_fold0` .. `cv_fold4` or similar.
**Never report a single fold's number as the result** — always aggregate to mean±std across
all 5, the same way the README's own 5-fold result (0.761 ± 0.021) was produced.
`collect_runs.py` does this aggregation automatically when it detects a `cv_fold*` naming
pattern; if run names differ, do it by hand rather than picking the best-looking fold.

## Comparing runs fairly

Before comparing two runs' metrics, check `args.yaml` for both — a difference in `imgsz`,
`batch`, `epochs`, or `MAX_NEGATIVES` (see `pcb-port-to-batch/references/script-inventory.md`'s
note on `gmo_detr_full_pipeline.py` setting this twice) can explain a metric gap on its own,
independent of any real modeling difference. Only attribute a gap to the intended variable
(resolution, architecture, pretraining) once everything else in `args.yaml` matches.
