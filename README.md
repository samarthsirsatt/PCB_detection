# PCB Defect Detection

Detecting manufacturing defects on **assembled PCBs (PCBA)** — from the imaging front-end (camera, resolution, working distance) through a full deep-learning detection pipeline (YOLO → RT-DETR → GMO-DETR).


---

## Overview

The project has two connected halves:

1. **Imaging front-end** — the optics engineering that decides whether a defect is even *physically resolvable* by the camera: defect sizes, the pixels-per-mm requirement, camera/sensor choice, working distance, depth of field, and capture strategy (full-board vs. component-level patches). This determines the ceiling on everything the detection model can achieve.

2. **Deep-learning detection** — building and training defect detectors on the annotated dataset, progressing from a YOLO baseline through RT-DETR to a from-scratch reimplementation of **GMO-DETR**, the current state of the art for PCBA defect detection.

---

## Dataset

The custom dataset consists of **~550 annotated defect images** across **12 defect classes**, plus **~1,200 defect-free (negative) images**, captured as zoomed-in views of assembled PCB components.

**12 classes:** Component Crack, Component Damage, Component Liftup, Component Missing, Component No Solder, Component Solder Dry, LED Damage, Polarity Wrong, RYB Wrong Sequence, Solder Ball, Solder Short, Tombstone.

Annotations were created in **CVAT** and exported in **COCO** format, then converted to YOLO format with a stratified (rarest-class-aware) train/val split.

### 📂 Dataset & Annotations Download

>
> The dataset images, defect-free negatives, and COCO annotation files are hosted on Google Drive:
>
> **Download:** `https://drive.google.com/drive/folders/1Uo_wORt9GAgkRp31MNxXnj2Sq2_PDcMs?usp=sharing`  
>
---



## Deep-Learning Detection

### Annotation

~550 defect images annotated in **CVAT** with bounding boxes + class labels, exported as COCO.

### YOLO Baseline & Transfer Learning

- **Baseline:** YOLO11s trained on the expanded ~550-image dataset (plus negatives). Substantial improvement over an earlier ~92-image run.
- **Transfer-learning study:** tested whether pretraining on the public **SolDef_AI** solder-defect dataset transfers better than standard ImageNet/COCO initialization.
  - **Finding:** SolDef_AI pretraining **did not help** — it reduced mAP@0.5 by ~10 points, with the drop near-uniform across classes (including solder classes). The likely cause is that SolDef_AI is small and narrow, so pretraining over-specialized the backbone and displaced the broad, general-purpose features that ImageNet initialization provides.

### 8-Class Study: Resolution & Cross-Validation

Four rarest classes (≤ ~15 instances) made per-class numbers noisy, so this study restricts to the **8 well-populated classes** for a reliable benchmark.

- **8-class baseline (640px):** mAP@0.5 **0.757**, mAP@0.5:0.95 **0.524**.
- **Resolution sweep (640 / 960 / 1280px):** higher resolution **did not help** — 960px is within noise on mAP@0.5 and worse on mAP@0.5:0.95; 1280px is worse on both. **640px is the right choice.**
- **5-fold cross-validation:** mAP@0.5 **0.761 ± 0.021**, mAP@0.5:0.95 **0.500 ± 0.020**. The ± is the real measurement uncertainty, confirming the baseline is stable and that the resolution differences are noise.

### RT-DETR

RT-DETR is the first **real-time end-to-end DETR** — it removes NMS (which costs YOLO both speed and accuracy) and uses an efficient hybrid encoder (AIFI + CCFF) to stay fast.

- **Config:** `rtdetr-l` (HGNetv2 backbone), 640px, 150 epochs, batch 8, AdamW, lr 1e-4 — identical data, split and augmentation as the YOLO baseline.
- **Results (8-class):** mAP@0.5 **0.770**, mAP@0.5:0.95 **0.556**, P 0.807, R 0.749 — beats YOLO on every metric, most clearly on the stricter mAP@0.5:0.95 (tighter boxes).
- **Cost:** ~5.8 it/s training (vs YOLO's 6.5, only ~11% slower) and ~33.6 ms inference (~30 FPS) — transformer accuracy at near-YOLO cost.

### GMO-DETR (from scratch)

**GMO-DETR** is a lightweight RT-DETR variant purpose-built for PCBA defects and is the current **state of the art** on the PCBA-DET benchmark (98.27% mAP@0.5 with 39.4% fewer parameters than RT-DETR).

**The authors released no code**, so the entire architecture was **reimplemented from scratch** from the paper's equations and figures:

- **GMONet** — GhostConv stem + dual-path MambaOut (DMambaOut) backbone
- **CAFF** — context-aware feature fusion (RepNCSPELAN4 + Context Anchor Attention)
- **TAIFI** — token-statistics attention (TSSA) replacing self-attention in AIFI
- **GSConv** — group-shuffle convolutions throughout the neck

Each module was cross-checked against its original source (MambaOut, PKINet, ToST) and integrated into the Ultralytics RT-DETR framework via custom module registration and a full model definition.

- **Config:** 640px, 200 epochs, batch 8, AdamW, lr 1e-4 (matching the paper's Table 2).
- **Results (12-class):** mAP@0.5 **0.534**, mAP@0.5:0.95 **0.350** — below the published figure because the paper trains on **6,384 images** while this run used **441**. A lightweight transformer with four novel modules is highly data-hungry; the architecture is proven and trains stably, but the bottleneck is data.

---

## Results Summary

**8-class dataset (our data, like-for-like):**

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|:-------:|:------------:|:---------:|:------:|
| YOLO11s (baseline) | 0.757 | 0.524 | 0.799 | 0.743 |
| RT-DETR | **0.770** | **0.556** | **0.807** | **0.749** |

**GMO-DETR (our data, 12-class):** mAP@0.5 0.534 · mAP@0.5:0.95 0.350 · P 0.682 · R 0.408
*(data-limited; only 441 training images)*

**GMO-DETR (published, PCBA-DET, 6,384 images):** mAP@0.5 **98.27%** · 12.05M params · 40.6 GFLOPs — state of the art.

---

## References

1. **RT-DETR** — Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J. *"DETRs Beat YOLOs on Real-time Object Detection."* CVPR 2024.
   https://doi.org/10.1109/CVPR52733.2024.01605

2. **GMO-DETR** — Su et al. *"GMO-DETR: a lightweight transformer for accurate and efficient PCBA defect detection."* Journal of Real-Time Image Processing, 2026, 23:83.
   https://doi.org/10.1007/s11554-026-01863-7


---

> **Follow-up study:** see [`OPTIMIZATION_STUDY.md`](OPTIMIZATION_STUDY.md) — an RT-DETR tuning pass (Aug 2026) that re-measures the numbers above against a proper noise floor and revises two of the conclusions on this page.
