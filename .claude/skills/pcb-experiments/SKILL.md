---
name: pcb-experiments
description: Extract, compare, and report results from completed PCB defect-detection training runs. Use when reading an Ultralytics runs/ directory (results.csv, args.yaml, weights/best.pt, confusion matrices), pulling mAP@0.5 / mAP@0.5:0.95 / precision / recall out of a finished job, aggregating 5-fold cross-validation folds into mean±std, comparing a new run against the published README baselines (YOLO11s 0.757/0.524, RT-DETR 0.770/0.556, GMO-DETR 0.534/0.350), or deciding whether a difference between two runs is real or within noise.
---

# PCB experiment results

Nothing here touches SLURM or the training scripts — it only reads finished output. No cluster
dependency at all; this is the one skill that also works fine off-cluster if results are copied
somewhere else for review.

## When NOT to use this skill

- The run hasn't finished / you need to check job status → `prajna-jobs`.
- You're about to submit or edit a training script → `pcb-port-to-batch` / `prajna-jobs`.

## The single most important fact here: the noise floor

From this project's own 5-fold cross-validation (README): **mAP@0.5 varies ±0.021 and
mAP@0.5:0.95 varies ±0.020 run-to-run, on the same model and data, from random seed and split
variation alone.** Two runs differing by less than roughly 0.04 are not distinguishable from
noise. This is exactly why the README concluded the 960px/1280px resolution sweep was noise,
not a real effect — 960px was *higher* than 640px on mAP@0.5 but within this band.

**Before reporting "run A beat run B," check whether the gap exceeds ~0.04.** If it doesn't,
say so explicitly rather than treating a noisy delta as a finding. Full baseline numbers and
the reasoning: `references/baselines.md`.

## Reading a run

Every Ultralytics run directory contains:
```
runs/<name>/
  results.csv        # per-epoch metrics; the LAST row is the final result
  args.yaml           # every hyperparameter the run was actually launched with
  weights/best.pt     # best checkpoint by the configured metric
  weights/last.pt     # final epoch's checkpoint (needed to resume)
  confusion_matrix.png, PR_curve.png, ...
```
`args.yaml` matters as much as `results.csv` — a surprising metric is often explained by a
hyperparameter that silently differed (batch size, imgsz, a `MAX_NEGATIVES` that got set twice
and the second value won — see `pcb-port-to-batch/references/script-inventory.md`'s note on
`gmo_detr_full_pipeline.py`).

## Aggregating

`scripts/collect_runs.py` walks `$PCB_RUNS_ROOT/*/` (see `PRAJNA_CONFIG.md` §8 for where that
is), reads the last row of `results.csv` plus `args.yaml`, and emits a markdown table of run
name / imgsz / batch / epochs / mAP50 / mAP50-95 / P / R / best-epoch / wall-time. Directories
matching `cv_fold*` are aggregated into mean±std automatically rather than listed individually
— the standalone number for a single fold is not the number to report, only the aggregate is.

This script is read-only and purely a convenience — every value it prints can be read directly
from `results.csv`/`args.yaml` by hand if you'd rather not run it.

## Comparing against baselines

`references/baselines.md` has the README's verified numbers for YOLO11s, RT-DETR, and
GMO-DETR, plus the interpretation of the noise floor above. Layout of what "good" looks like
for this project: `references/run-layout.md` also covers which files matter for a writeup and
which are safe to discard once you've captured the metrics.

## Escape hatch

If a run's output doesn't fit the standard Ultralytics layout (a custom logging path, a script
that writes metrics somewhere else), read it directly — nothing here requires the standard
layout, `collect_runs.py` is just a shortcut when it applies.
