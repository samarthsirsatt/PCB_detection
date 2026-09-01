# PCB Defect Detection — Combined Optimization Progress Report

## 1. Overview

This report summarizes the optimization and comparison work performed on the existing PCB defect-detection pipeline.

The study focuses on the **8-class PCB defect dataset** and compares:

- YOLO11s
- RT-DETR
- Optimized RT-DETR
- GMO-DETR
- Optimized GMO-DETR

The main objective was to determine which detector is most suitable for the current dataset and to identify the main factors limiting detection performance.

---

## 2. Dataset and Evaluation

The experiments use the existing 8-class PCB defect dataset.

- 508 images are used in the main 8-class experiments.
- The dataset contains an average of **1.11 objects per image**.
- 474 of 508 images contain exactly one object.
- The validation set contains 107 defect instances across the 8 classes.
- The dataset does not contain board IDs, so a completely board-held-out split cannot currently be guaranteed.

Because the dataset is relatively small, individual training runs can show noticeable variation. Important findings were therefore checked with repeated runs where possible.

---

## 3. Initial YOLO11s and RT-DETR Comparison

The original work did not initially use identical evaluation procedures for YOLO11s and RT-DETR. The comparison was therefore repeated using the same 5-fold evaluation protocol.

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| YOLO11s | **0.761 ± 0.021** | **0.500 ± 0.020** |
| RT-DETR | 0.755 ± 0.034 | 0.494 ± 0.036 |

### Observation

Under the same evaluation procedure, the current data does **not establish a meaningful accuracy advantage of RT-DETR over YOLO11s**.

---

## 4. RT-DETR Optimization Study

Several targeted changes were tested on RT-DETR.

| Experiment | mAP@0.5 | mAP@0.5:0.95 | Change vs baseline |
|---|---:|---:|---:|
| Baseline | 0.743 | 0.479 | — |
| **30 object queries** | **0.789** | **0.567** | **+0.046 / +0.088** |
| 30 queries — repeat | **0.783** | **0.531** | **+0.040 / +0.052** |
| **1280 px input** | **0.787** | **0.520** | **+0.044 / +0.041** |
| 300 defect-free images | 0.765 | 0.518 | +0.022 / +0.039 |
| Vertical flip disabled | 0.761 | 0.529 | +0.018 / +0.050 |
| Rectangular training | 0.744 | 0.513 | +0.001 / +0.034 |
| 600 defect-free images | 0.718 | 0.482 | −0.026 / +0.003 |
| 900 defect-free images | 0.699 | 0.489 | −0.044 / +0.010 |
| 1,195 defect-free images | 0.670 | 0.446 | −0.073 / −0.033 |
| All four changes combined | 0.762 | 0.453 | +0.019 / −0.026 |

### Main RT-DETR finding

The strongest and most reproducible improvement was reducing the RT-DETR decoder object-query count from **300 to 30**.

Two independent runs achieved:

- 0.789 mAP@0.5
- 0.783 mAP@0.5

giving a two-run mean of:

**0.786 mAP@0.5**

The reduced-query configuration also showed more stable training and slightly lower computational cost.

### Resolution finding

The earlier resolution experiment changed batch size together with image resolution, so the effect of resolution was not isolated.

A corrected experiment with fixed batch size found:

**1280 px → 0.787 mAP@0.5**

compared with the 640-pixel baseline of 0.743, an improvement of approximately **+0.044 mAP@0.5**.

### Negative-image finding

Adding defect-free images helped only up to a point. Approximately **300 defect-free images** gave the best tested result. Increasing the number to 600, 900 or all 1,195 images reduced performance.

### Augmentation finding

Vertical flipping was removed because **Polarity Wrong** and **RYB Wrong Sequence** are orientation-dependent classes. The change mainly improved the stricter mAP@0.5:0.95 metric.

---

## 5. RT-DETR Localization Analysis

A separate classification experiment was performed using ground-truth defect crops.

| Task | Result |
|---|---:|
| Classification accuracy with perfect boxes | **90.7%** |
| RT-DETR baseline mAP@0.5 | **74.3%** |

This indicates that the main remaining difficulty is likely **locating and tightly bounding the defect**, rather than simply distinguishing one defect class from another.

---

## 6. GMO-DETR Study

GMO-DETR was evaluated on the **same 8-class dataset, same split and same augmentation framework** as the RT-DETR study.

The aim was to determine whether the more specialized GMO-DETR architecture could become competitive after reasonable optimization.

### GMO-DETR baseline

| Metric | GMO-DETR | RT-DETR baseline |
|---|---:|---:|
| mAP@0.5 | **0.440** | 0.743 |
| mAP@0.5:0.95 | **0.252** | 0.479 |
| Precision | 0.415 | 0.713 |
| Recall | 0.452 | 0.724 |
| Parameters | **17.9 M** | 32.0 M |
| GFLOPs | **58.0** | 105.4 |
| Inference | 9.3 ms/img | 5.8 ms/img |

GMO-DETR is considerably lighter in terms of parameters and FLOPs, but its accuracy is much lower on the current dataset.

---

## 7. GMO-DETR Optimization Experiments

The main targeted changes were:

- input resolution
- object-query count
- vertical flipping
- defect-free images
- stride-4 feature head for the small-object class Solder Ball

| Configuration | Mean / Result mAP@0.5 | mAP@0.5:0.95 | Change vs baseline |
|---|---:|---:|---:|
| Baseline | 0.440 | 0.252 | — |
| Vertical flip off | **0.478** | 0.262 | +0.038 |
| Flip off + 300 negatives + P2 | **0.475** | **0.275** | +0.035 |
| 1280 px | 0.451 | 0.224 | +0.011 |
| 300 negatives | 0.447 | 0.247 | +0.007 |
| Flip off + 300 negatives | 0.430 | 0.266 | −0.010 |
| Flip off + P2 | 0.410 | 0.235 | −0.030 |
| P2 head only | 0.403 | 0.248 | −0.037 |
| 30 object queries | 0.387 | 0.205 | −0.053 |

The best repeated GMO-DETR configuration was **flip-off + 300 negatives + P2**, with approximately **0.475 mAP@0.5**. However, this improvement is not established as statistically meaningful because GMO-DETR has much larger run-to-run variation.

---

## 8. GMO-DETR Stability Finding

Three identical GMO-DETR baseline runs produced:

- 0.489
- 0.390
- 0.133

The observed spread was approximately **±0.069 mAP@0.5**.

Two of 18 GMO-DETR runs also collapsed to approximately 0.13.

This means that single-run improvements are not reliable enough to be treated as confirmed gains.

The 30-query change illustrates this clearly:

| Model | Baseline | 30 queries |
|---|---:|---:|
| RT-DETR | 0.743 | **0.789** |
| GMO-DETR | 0.440 | **0.387** |

Therefore, the RT-DETR query optimization does **not** transfer automatically to GMO-DETR.

---

## 9. GMO-DETR and Small Objects

Solder Ball is the main small-object class in the dataset, so a stride-4 (P2) feature level was tested.

The P2 head did not produce a proven improvement in overall mAP, but Solder Ball detection became more consistent:

- P2: mean AP@0.5 ≈ **0.267**
- Without P2: mean AP@0.5 ≈ **0.253**

This result is suggestive but not statistically conclusive.

---

## 10. Final Combined Comparison

| Model | mAP@0.5 | mAP@0.5:0.95 | Main observation |
|---|---:|---:|---|
| YOLO11s, 5-fold CV | 0.761 ± 0.021 | 0.500 ± 0.020 | Strong baseline |
| RT-DETR, 5-fold CV | 0.755 ± 0.034 | 0.494 ± 0.036 | Similar to YOLO11s |
| **Optimized RT-DETR (30 queries)** | **0.786** | **0.549** | **Best result** |
| GMO-DETR | 0.440 ± 0.069 | 0.252 | High run-to-run variability |
| Optimized GMO-DETR | 0.475 ± 0.059 | 0.275 | Small, unconfirmed improvement |

---

## 11. Overall Conclusions

### Best-performing model

The best current detector is:

**RT-DETR with 30 object queries**

with a two-run mean of:

**0.786 mAP@0.5**

and:

**0.549 mAP@0.5:0.95**

### YOLO11s vs standard RT-DETR

The same 5-fold evaluation does not show a clear accuracy advantage for standard RT-DETR over YOLO11s.

### GMO-DETR

GMO-DETR is considerably lighter than RT-DETR in parameters and FLOPs, but its detection performance is substantially lower on this dataset.

Its best repeated configuration reaches approximately **0.475 mAP@0.5**, compared with **0.786 for optimized RT-DETR**.

The gap is:

**0.311 mAP@0.5**

which is much larger than the measured run-to-run variation.

### Why GMO-DETR underperforms

The current experiments suggest that the main issue is not simply the GMO-DETR architecture.

GMO-DETR is trained **from scratch** on approximately 441 training images, whereas RT-DETR starts from pretrained weights. The available dataset is therefore likely too small for GMO-DETR to demonstrate its full potential.

### Main technical bottleneck

The localization experiment indicates that finding the correct defect location is a larger challenge than classifying an already-correctly-cropped defect.

This points toward improving:

- image resolution
- defect localization
- small-object handling
- dataset quality and scale
- training stability

rather than simply introducing increasingly complex architectures.

---

## 12. Current Research Position

Based on the experiments completed so far:

**Optimized RT-DETR is the preferred detector for the current dataset.**

GMO-DETR has been tested and optimized with several targeted changes, but no optimization has produced a reliable improvement large enough to make it competitive with optimized RT-DETR.

The next stage should therefore focus on **localization and the imaging/data pipeline**, while GMO-DETR can be revisited later if substantially more training data or suitable pretraining becomes available.

The most immediate technical experiment is to investigate **small-object/localization improvements in RT-DETR**, particularly for Solder Ball, and then connect these results with the planned PCB imaging setup.
