# RT-DETR Optimization Study

August–October 2026 · 27 training runs · Prajna HPC cluster (NVIDIA L40S / A40 / A100)

---

## Summary

Reducing RT-DETR's object query count from 300 to 30 raised mAP@0.5 from **0.743 to 0.789** on the 8-class dataset, and mAP@0.5:0.95 from **0.479 to 0.567**. The result was confirmed on two independent training runs, and the model also trained more stably. This was the largest of six changes tested.

A follow-up in October added each of the other beneficial changes to the 30-query model individually, with two runs each. None improved on 30 queries alone, and higher resolution and additional defect-free images both made it measurably worse. **30 queries at 640 px, with all other settings at baseline, remains the recommended configuration.**

The study also re-examined two conclusions from the earlier work. Neither holds up:

- RT-DETR does **not** measurably outperform YOLO11s once both models are cross-validated.
- The earlier finding that higher input resolution does not help came from an experiment that changed two variables at once. Corrected, resolution helps considerably.

---

## Part 1 — Rechecking the earlier results

### Measurement uncertainty on this dataset

The validation set contains 107 defect instances across 8 classes, so individual scores carry substantial uncertainty. Two independent estimates agree closely:

- **Across data splits:** the published YOLO11s 5-fold cross-validation gives ±0.021 on mAP@0.5.
- **Within a single run:** tracking mAP@0.5 across consecutive epochs of one training run gives a standard deviation of ±0.022, with a peak-to-trough swing of 0.076.

In practice, **any difference below roughly 0.02 mAP cannot be distinguished from ordinary run-to-run variation.** This threshold is used throughout the report.

### RT-DETR versus YOLO11s

The earlier comparison placed a single RT-DETR run (0.770) against YOLO's cross-validated mean (0.757). These are not equivalent measurements. Running RT-DETR through the same 5-fold protocol gives:

| Model | mAP@0.5 | mAP@0.5:0.95 |
|-------|:---:|:---:|
| YOLO11s | 0.761 ± 0.021 | 0.500 ± 0.020 |
| RT-DETR | 0.755 ± 0.034 | 0.494 ± 0.036 |

The two distributions overlap almost completely, and RT-DETR's mean is slightly lower. The original 0.013 gap is smaller than the uncertainty in either measurement, so **no performance difference between the two architectures is established by this data.** RT-DETR's speed advantage (NMS-free inference, ~30 FPS) is unaffected and remains a valid reason to prefer it.

### The resolution sweep

The earlier sweep reduced batch size as it raised resolution — 640 px at batch 16, 960 px at batch 12, 1280 px at batch 6 — because the 16 GB Kaggle T4 could not hold larger batches at higher resolution. Batch size and resolution both affect accuracy, so the experiment could not separate them.

Repeating the sweep on a 47.7 GB L40S with **batch size fixed at 8** reverses the conclusion: 1280 px scores 0.787, a gain of 0.044 over the 640 px baseline. This is consistent with the source images being natively 1280×720 — training at 640 px square discards roughly 44% of the frame to letterbox padding.

---

## Part 2 — What was changed and why

Six changes were tested, each motivated by a measured property of the dataset rather than general tuning practice.

| Property measured | Value | Change tested |
|---|---|---|
| Objects per image | 1.11 (474 of 508 images contain exactly one) | Reduce object queries from 300 to 30 |
| Native image size | 1280×720 for 550 of 551 images | Train at 1280 px instead of 640 px |
| Defect-free images used | 120 of 1,195 available | Increase to 300, 600, 900, and all 1,195 |
| Two classes defined by orientation | Polarity Wrong, RYB Wrong Sequence | Disable vertical-flip augmentation |
| Letterbox padding at 640 px | ~44% of the frame | Enable rectangular training |

### Results

All runs use RT-DETR-l on the 8-class dataset with identical data, splits, and batch size. The baseline is a reproduction of the existing `rtdetr.py` configuration.

| Configuration | mAP@0.5 | mAP@0.5:0.95 | Change vs baseline |
|---|:---:|:---:|:---:|
| Baseline (no changes) | 0.743 | 0.479 | — |
| **Object queries reduced to 30** | **0.789** | **0.567** | **+0.046** |
| Object queries reduced to 30 (repeat run) | 0.783 | 0.531 | +0.040 |
| **Input resolution raised to 1280 px** | **0.787** | 0.520 | **+0.044** |
| Defect-free images increased to 300 | 0.765 | 0.518 | +0.022 |
| Vertical flipping disabled | 0.761 | 0.529 | +0.018 |
| Rectangular training (no letterbox) | 0.744 | 0.513 | +0.001 |
| Defect-free images increased to 600 | 0.718 | 0.482 | −0.026 |
| Defect-free images increased to 900 | 0.699 | 0.489 | −0.044 |
| Defect-free images increased to 1,195 (all) | 0.670 | 0.446 | −0.074 |
| All four improvements applied together | 0.762 | 0.453 | +0.019 |

---

## Part 3 — Discussion

### Reducing object queries is the main result

RT-DETR searches every image for up to 300 objects by default. This dataset averages 1.11 objects per image, so almost all of the model's training signal goes into teaching 299 queries to report "nothing here." Reducing the decoder to 30 queries — with the pretrained weights transferred intact, 926 of 941 tensors carrying over unchanged — improved every metric:

- mAP@0.5 rose 0.046 and mAP@0.5:0.95 rose 0.088, both well beyond measurement uncertainty.
- A second run with a different random seed reproduced the gain (+0.040), confirming it is a real effect.
- Epoch-to-epoch variation halved, from ±0.022 to ±0.012. The model trains more consistently, not only more accurately.
- Computation dropped slightly, from 105.4 to 100.8 GFLOPs.

This is the recommended configuration going forward.

### Defect-free images help only in moderation

Adding defect-free images to the training mix improved results at 300 images but degraded them at 600 and beyond, falling well below baseline when all 1,195 were used. As the proportion of empty images grows, the model increasingly learns to predict nothing at all. **300 is the measured optimum**, not the maximum available.

### Removing vertical flipping helps localization accuracy

Vertical flipping was being applied to Polarity Wrong and RYB Wrong Sequence, two classes whose labels depend on component orientation — flipping these images makes the label incorrect. Disabling it moved mAP@0.5 only within uncertainty (+0.018), but improved mAP@0.5:0.95 by 0.050, indicating tighter boxes. No class performed worse.

### The improvements do not combine

Applying the four beneficial changes together scored 0.762, **below the 0.789 achieved by reducing object queries alone.** The changes interact rather than accumulate. The most likely explanation is that a 30-query decoder requires a different learning-rate schedule at 1280 px than the one tuned at 640 px. Establishing a combined configuration would require a joint search rather than simple stacking.

The October follow-up (Part 6) tested each change against the 30-query model separately and found that the conflict is not specific to 1280 px: additional defect-free images also hurt the 30-query model at 640 px, where the learning-rate explanation does not apply.

### Two runs failed and were caught by repetition

Two experiments initially produced very poor scores — 0.388 and 0.644 — that appeared to be strong negative findings. Both showed epoch-to-epoch variation an order of magnitude above normal (0.179 and 0.258 against a typical 0.02), indicating unstable training rather than a genuine effect. Repeating both with a different random seed produced 0.699 and 0.740, within the normal range. The table above reports the corrected values.

This has a practical consequence: **on a dataset this small, a single training run's score should not be reported without either a repeated run or its epoch-variation figure.**

### The remaining error is in localization, not classification

Since 93% of images contain exactly one object, a classifier was trained on ground-truth crops to establish how much of the error is due to confusing one defect type with another:

| | Score |
|---|:---:|
| Classification accuracy, given a perfect box | 0.907 |
| Detection mAP@0.5, baseline | 0.743 |
| Difference | 0.164 |

The classes are highly distinguishable — 90.7% top-1 accuracy with only 4 to 35 examples per class in validation. The 16-point gap therefore reflects the difficulty of **finding and bounding** defects, not telling them apart. This explains why changes affecting query allocation and resolution produced gains, and indicates that effort directed at improved feature extraction or classification capacity would address the wrong limitation.

---

## Part 4 — Revised results

| Configuration | mAP@0.5 | mAP@0.5:0.95 |
|---|:---:|:---:|
| YOLO11s, 5-fold cross-validation | 0.761 ± 0.021 | 0.500 ± 0.020 |
| RT-DETR, 5-fold cross-validation | 0.755 ± 0.034 | 0.494 ± 0.036 |
| **RT-DETR with 30 object queries** (mean of two runs) | **0.786** | **0.549** |
| RT-DETR with 30 queries + any other change (best: no vertical flip, mean of two runs) | 0.780 | 0.515 |
| Classification accuracy given perfect localization | 0.907 | — |

---

## Part 5 — Recommended next steps

1. **Adopt 30 object queries as the RT-DETR default.** It is the only change confirmed across two runs, and it improves both accuracy and training stability.

2. ~~**Investigate why the improvements conflict.**~~ *Done in October — see Part 6.* Each change was added to the 30-query model separately; none helped. Whether the 1280 px result could be recovered with a retuned learning-rate schedule remains untested, but given that defect-free images also conflict at 640 px, this is no longer the leading explanation.

3. **Report every future result with a repeated run.** Two of ten experiments here would have been misreported as findings without a repeat.

4. **Add a stride-4 detection head for Solder Ball only.** It is the one genuinely small-object class in the dataset (0.61% median area against 1.87% or more for every other class) and the class that gained most from higher resolution.

5. **Re-run GMO-DETR on the 8-class subset.** Its current 12-class score of 0.534 cannot be compared with any 8-class figure, since the four additional classes have 6 to 15 instances each and depress the mean regardless of architecture.

6. **Examine Component No Solder.** It is the weakest class in every configuration tested (0.28 to 0.39 mAP@0.5:0.95) with no obvious explanation from size or frequency, suggesting possible annotation inconsistency against Component Solder Dry.

7. **Resolve the unreproduced mAP@0.5:0.95 figure.** The baseline reproduced closely on mAP@0.5 (0.766 against the published 0.770) but not on the stricter metric (0.479 against 0.556). Two untested explanations remain: a newer Ultralytics version (8.4.127), and the original run's use of two GPUs, which changes effective batch statistics.

---

## Part 6 — Follow-up: adding changes to the 30-query model (October 2026)

The combined run in Part 3 stacked three changes onto 30 queries at once, so it could not show which of them caused the drop. Each was therefore added to the 30-query model on its own, with two runs (seeds 42 and 1) per configuration. All other settings are unchanged from Part 2.

| Configuration | mAP@0.5 (seed 42 / seed 1) | mAP@0.5 mean | mAP@0.5:0.95 mean | Change vs 30 queries |
|---|:---:|:---:|:---:|:---:|
| **30 queries (reference)** | 0.789 / 0.783 | **0.786** | **0.549** | — |
| 30 queries, vertical flipping disabled | 0.797 / 0.763 | 0.780 | 0.515 | −0.006 / −0.034 |
| 30 queries, 1280 px | 0.766 / 0.724 | 0.745 | 0.491 | −0.041 / −0.058 |
| 30 queries, 300 defect-free images | 0.759 / 0.696 | 0.728 | 0.473 | −0.058 / −0.076 |

**No change improves on 30 queries alone.** Higher resolution and additional defect-free images each lower both metrics by two to three times the measurement uncertainty, consistently across both runs. Disabling vertical flipping leaves mAP@0.5 within uncertainty but lowers mAP@0.5:0.95 on both runs, so the tighter boxes it produced at 300 queries do not carry over.

This accounts for the combined result in Part 3: all three changes it added are individually neutral or harmful once queries are reduced.

**Why the changes stop helping.** The likeliest explanation is that the changes and the query reduction address overlapping problems. Additional defect-free images and the 30-query decoder both push the model toward predicting fewer objects; together they over-suppress detections, which matches mean recall falling from 0.76 to 0.69 in the defect-free-image runs. This is an interpretation, not a tested result.

**Training stability.** All six runs trained normally, with epoch-to-epoch variation of 0.006 to 0.029 in mAP@0.5, so none of these results reflects a failed run of the kind described in Part 3. One run (30 queries, 1280 px, seed 1) reports a low precision of 0.585 because its saved checkpoint fell on a weaker epoch; its peak mAP@0.5 during training was 0.777, which still trails the reference.

---

## Appendix — Reproducing the experiments

```
experiments/
  exp_common.py    COCO-to-YOLO conversion and split logic, shared with rtdetr.py
  exp_rtdetr.py    a single configurable run; all settings are environment variables
  exp_classify.py  the classification-accuracy experiment
  collect.py       aggregates results into the tables above
  run_exp.sbatch   SLURM job template
```

Each run is one command. For example, the main result:

```bash
EXP_NAME=nq30 NUM_QUERIES=30 sbatch experiments/run_exp.sbatch
```

Run directories map to the table as follows: `base` (baseline), `nq30` and `nq30_s1` (30 queries, two runs), `res1280` (1280 px), `neg300` through `neg1195` (defect-free image counts), `flipud0` (no vertical flip), `rect` (rectangular training), `best` (combined), `cv0`–`cv4` (cross-validation folds), `cls_ceiling` (classification test). The Part 6 follow-up runs are `nq30_fu0`, `nq30_r1280` and `nq30_neg300`, each with an `_s1` repeat, for example:

```bash
EXP_NAME=nq30_r1280 NUM_QUERIES=30 IMGSZ=1280 SEED=42 sbatch experiments/run_exp.sbatch
```

Per-epoch curves, confusion matrices and precision-recall curves for all 27 runs are preserved under `runs/experiments/`. Cluster configuration notes are in `.claude/skills/SITE_AMENDMENTS.md`.
