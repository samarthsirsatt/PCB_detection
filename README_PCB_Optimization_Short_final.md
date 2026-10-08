# PCB Defect Detection — RT-DETR Optimization Study

## Objective

I re-examined the existing PCB defect-detection pipeline and evaluated targeted changes to the RT-DETR training setup on the 8-class dataset. The main goal was to determine which changes provide a measurable improvement and to identify the current bottleneck in detection performance.

## Dataset

The experiments use the existing 8-class PCB defect dataset.

- 508 images are used in the main experiment.
- The dataset contains an average of **1.11 objects per image**.
- **474 of 508 images contain exactly one object**.
- The validation set contains 107 defect instances across the 8 classes.
- Because the original dataset does not contain board IDs, a completely board-held-out split cannot currently be constructed.

## Main Results

### Overall experiment summary

I tested several targeted changes to the RT-DETR training configuration. All experiments in this table use the same 8-class dataset and baseline evaluation setup unless noted otherwise.

| Experiment | mAP@0.5 | mAP@0.5:0.95 | Change vs baseline |
|---|---:|---:|---:|
| Baseline RT-DETR | 0.743 | 0.479 | — |
| **30 object queries** | **0.789** | **0.567** | **+0.046 / +0.088** |
| 30 object queries — repeat run | 0.783 | 0.531 | +0.040 / +0.052 |
| **1280-pixel input** | **0.787** | **0.520** | **+0.044 / +0.041** |
| 300 defect-free images | 0.765 | 0.518 | +0.022 / +0.039 |
| Vertical flip disabled | 0.761 | 0.529 | +0.018 / +0.050 |
| Rectangular training | 0.744 | 0.513 | +0.001 / +0.034 |
| 600 defect-free images | 0.718 | 0.482 | −0.026 / +0.003 |
| 900 defect-free images | 0.699 | 0.489 | −0.044 / +0.010 |
| 1,195 defect-free images | 0.670 | 0.446 | −0.073 / −0.033 |
| All four changes combined | 0.762 | 0.453 | +0.019 / −0.026 |

### 1. RT-DETR object queries

The original RT-DETR configuration uses **300 object queries**. Since this dataset contains an average of **1.11 objects per image**, I tested whether a smaller query budget would be better suited to the task.

| Configuration | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| 300 queries | 0.743 | 0.479 |
| **30 queries** | **0.789** | **0.567** |
| 30 queries — repeat | **0.783** | **0.531** |

Compared with the baseline, the two-query-budget experiments improved mAP@0.5 by **0.046** and **0.040** respectively. The two-run mean is **0.786 mAP@0.5**. Training was also more stable and computation decreased slightly.

### 2. Input resolution

The earlier resolution experiment changed both resolution and batch size, so I repeated it with batch size fixed at 8.

| Input resolution | mAP@0.5 | Change vs 640 |
|---|---:|---:|
| 640 px | 0.743 | — |
| **1280 px** | **0.787** | **+0.044** |

The result indicates that higher-resolution input can improve defect detection on these images. The source images are predominantly 1280×720, so reducing them to 640 px can remove useful spatial information.

### 3. Number of defect-free images

I tested how the amount of negative data affects performance.

| Defect-free images | mAP@0.5 | Change vs baseline |
|---:|---:|---:|
| 120 | 0.743 | — |
| **300** | **0.765** | **+0.022** |
| 600 | 0.718 | −0.026 |
| 900 | 0.699 | −0.044 |
| 1,195 | 0.670 | −0.073 |

This shows that adding negative examples helps up to a point, but adding a large number of defect-free images reduces performance. **300 was the best value tested.**

### 4. Data augmentation

Vertical flipping was disabled because the **Polarity Wrong** and **RYB Wrong Sequence** classes are orientation-dependent.

| Augmentation setting | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| Vertical flip enabled | 0.743 | 0.479 |
| **Vertical flip disabled** | **0.761** | **0.529** |

The change gave a modest **+0.018 mAP@0.5** improvement and a larger **+0.050 mAP@0.5:0.95** improvement.

### 5. Rectangular training

I also tested training without square letterbox padding.

| Configuration | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| Baseline | 0.743 | 0.479 |
| Rectangular training | 0.744 | 0.513 |

The change in mAP@0.5 was only **+0.001**, so this was not a meaningful improvement on its own.

### 6. YOLO11s vs RT-DETR under the same evaluation

I re-checked the earlier comparison using the same 5-fold evaluation procedure for both architectures.

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| YOLO11s | **0.761 ± 0.021** | **0.500 ± 0.020** |
| RT-DETR | 0.755 ± 0.034 | 0.494 ± 0.036 |

Under the same evaluation protocol, the current data does not establish a clear accuracy advantage for RT-DETR over YOLO11s.

### 7. Effect of combining improvements

The individual experiments show useful gains, but the gains cannot simply be added together.

| Configuration | mAP@0.5 | Change vs baseline |
|---|---:|---:|
| 30 queries only | **0.789** | **+0.046** |
| 1280 px only | **0.787** | **+0.044** |
| All four changes together | 0.762 | +0.019 |

This shows that the training settings interact with each other. The combined configuration was worse than the best individual change, so a joint parameter search would be needed before claiming that all improvements work together.

## Current Best Configuration

Based on the experiments completed so far, the most reliable RT-DETR configuration is:

- **30 object queries**
- Existing 8-class dataset
- Current training pipeline
- Repeated training runs for verification

The two 30-query runs give a mean of **0.786 mAP@0.5**.

## Current Direction

The results indicate that the main area for further improvement is **defect localization**.

The current investigation is therefore focused on:

1. Understanding the interaction between query count and input resolution.
2. Improving detection of small defects, particularly **Solder Ball**.
3. Ensuring that the train/validation/test split does not overestimate performance through component overlap.
4. Connecting the detector performance with the PCB imaging setup, including image resolution and field of view.
5. Evaluating whether the current RT-DETR approach is sufficient before introducing a more complex architecture such as GMO-DETR.

## Summary

The optimization study shows that the RT-DETR configuration can be improved substantially for the characteristics of this PCB dataset. The strongest confirmed change so far is reducing the object-query count from **300 to 30**, while the resolution experiment shows that higher-resolution input can also improve detection.

At the same time, the results show that simply changing the detector architecture is unlikely to solve the entire problem. The current evidence points toward **localization, image resolution, and dataset/evaluation quality** as the more important areas to investigate.
