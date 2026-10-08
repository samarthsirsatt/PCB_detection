# FOV vs Localization Analysis

**Question:** does the magnification an image was captured at (its field of view, FOV) change how
well our detector finds and bounds defects?

**Method (no retraining).** The already-trained **RT-DETR-l, 30 object queries**
(`runs/experiments/nq30/weights/best.pt`) was re-evaluated as-is on the *same* validation split it
was validated on (SEED=42, 8-class benchmark: 125 images = 101 with defects + 24 defect-free,
**107 ground-truth boxes**). Each image's FOV was read from its own EXIF `UserComment`
(`px/mm = 1280 / FOV`, see [`PCB_IMAGING_ANALYSIS.md`](PCB_IMAGING_ANALYSIS.md) §2). Images were
split into six FOV bins and the model was evaluated separately on each, at imgsz=640.

*Sanity check:* re-running the whole validation set reproduced the recorded result exactly —
**0.7898 / 0.5674** here vs **0.7892 / 0.5668** in `runs/experiments/nq30/summary.json`. The
evaluation is faithful.

---

## Main table — same model, evaluated per FOV bin

| FOV bin | px/mm | Images (defect + OK) | Defects | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–10 mm | 142–640 | 10 (9 + 1) | 10 | 0.917 | 0.645 | 0.715 | 0.917 |
| 10–15 mm | 91–128 | 20 (16 + 4) | 17 | 0.730 | 0.487 | 0.753 | 0.725 |
| 15–20 mm | 71–85 | 28 (21 + 7) | 21 | 0.773 | 0.507 | 0.763 | 0.847 |
| 20–25 mm | 64 | 22 (20 + 2) | 20 | 0.838 | 0.651 | 0.771 | 0.789 |
| 25–30 mm | 51 | 17 (12 + 5) | 13 | 0.701 | 0.580 | 0.923 | 0.580 |
| 30–40 mm | 32–43 | 28 (23 + 5) | 26 | 0.809 | 0.578 | 0.745 | 0.790 |
| **All (reference)** | 32–640 | **125 (101 + 24)** | **107** | **0.790** | **0.567** | 0.745 | 0.779 |

**The scores go up and down, but not in any order.** They do not fall as FOV widens (30–40 mm
scores 0.809, above the overall 0.790) and they do not rise as FOV narrows in a straight line
either. Each bin holds only **10–26 defects**, against a noise floor already measured at ±0.021 on
the *full* 107-instance set — so these bins are far noisier than that.

---

## Is the variation real, or just small samples?

Per-bin mAP is too noisy to answer this, so localization quality was also measured **per
ground-truth box** (107 points): for each true defect, the best IoU achieved by any prediction of
the correct class (conf ≥ 0.10). Higher IoU = tighter box.

| FOV bin | Defects | Median best IoU | Mean IoU | Found @ IoU 0.5 | Found @ IoU 0.75 |
|---|---:|---:|---:|---:|---:|
| 1–10 mm | 10 | 0.845 | 0.804 | 90% | 90% |
| 10–15 mm | 17 | 0.821 | 0.641 | 71% | 65% |
| 15–20 mm | 21 | 0.827 | 0.692 | 81% | 67% |
| 20–25 mm | 20 | 0.884 | 0.726 | 85% | 70% |
| 25–30 mm | 13 | 0.857 | 0.655 | 69% | 69% |
| 30–40 mm | 26 | 0.831 | 0.708 | 85% | 69% |

Two tests on those 107 points:

| Test | Result | Reading |
|---|---:|---|
| Spearman ρ (px/mm vs best IoU) | **−0.028** (n=107) | **no relationship** |
| Spearman ρ (FOV mm vs best IoU) | +0.028 (n=107) | no relationship |
| Spearman ρ (**defect size in px** vs best IoU) | **+0.294** (n=107, t=3.16, p≈0.002) | real, moderate |
| Spearman ρ (defect size in mm vs best IoU) | +0.249 (n=107, p≈0.01) | real, weaker |
| Permutation test: is the between-bin spread bigger than chance? | **p = 0.69** (hit rate), **p = 0.70** (mean IoU) | **no** |

The permutation test shuffled the FOV labels across the 107 boxes 20,000 times. Randomly assigned
bins produce a *larger* typical spread (0.241 hit rate) than the real bins do (0.208). **The
differences between FOV bins are indistinguishable from random sampling.**

---

## The three focus classes

Instance counts per bin are 1–7, so these are observations, not measurements.

| Class | Defects in val | Median IoU | Found @ 0.5 | ρ (px/mm vs IoU) | Note |
|---|---:|---:|---:|---:|---|
| **Solder Ball** | 8 | 0.763 | 88% | +0.33 (n=8) | Its one failure (IoU 0.00) is in the 25–30 mm bin, but the 30–40 mm bin — *wider* FOV, fewer px/mm — was found at IoU 0.88. One miss, no trend. |
| **Solder Short** | 30 | 0.782 | 73% | −0.17 (n=30) | Worst bin is **10–15 mm** (median IoU 0.430, 43% found) — a *high*-magnification bin at 98 px/mm. The opposite of what a resolution shortage would predict. |
| **Component No Solder** | 17 | **0.373** | **47%** | −0.03 (n=17) | Weakest class overall, and it is weak in **every** bin (IoU 0.19 → 0.90 with no ordering). Its boxes are mid-sized (median 155 px / 2.1 mm), so neither size nor FOV explains it. |

Component No Solder failing uniformly across all magnifications is consistent with the existing
annotation-consistency hypothesis (vs Component Solder Dry) rather than anything optical.

---

## Conclusion

**Does variable FOV appear to be affecting localization? — No, not in this data.**

Across 107 validation defects, px/mm shows **essentially zero** rank correlation with box quality
(ρ = −0.028), and a permutation test says the differences between FOV bins are **smaller than
chance would produce** (p = 0.69). What *does* correlate is how many pixels a defect occupies
(ρ = +0.294, p ≈ 0.002) — and that is set by the annotation, not by the magnification, since FOV and
box pixel-area are uncorrelated in this dataset (r = 0.03, `PCB_IMAGING_ANALYSIS.md` §5).

This **refines the earlier suggestion** in `PCB_IMAGING_ANALYSIS.md` §8 that scale inconsistency
might drive poor localization (ρ = −0.69). That figure came from only 8 class-level points and does
**not** survive this stronger 107-point per-instance test. The honest reading now: *defect pixel
size* is weakly linked to localization quality; *variable FOV itself* is not.

**Caveats.** One trained model, one split, 10–26 defects per bin, per-class counts as low as 1.
This can rule an effect *out* at the size a small dataset can detect; it cannot prove one is absent.
A per-bin difference smaller than roughly 0.1 mAP would be invisible here.

**Implication.** Scale normalisation (recommendation #1 of the imaging analysis) is now a **lower**
priority — the mechanism it targets does not show up in the measurements. Effort is better spent on
Component No Solder's annotation consistency and on defect pixel size (the stride-4 head), which is
the one variable that did show a real signal.

---

*Reproduce:* `imaging/fov_bins.py` (binning) → `imaging/fov_eval.py` (per-bin val) →
`imaging/fov_iou.py` (per-box IoU). Outputs: `imaging/val_fov_bins.csv`,
`imaging/fov_bin_results.{json,csv}` (includes per-class AP@0.5 and mAP@0.5:0.95 per bin),
`imaging/val_gt_iou.csv`. Run with `envs/pcb/bin/python`; evaluated on CPU, no training performed.*
