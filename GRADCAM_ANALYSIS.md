# Grad-CAM Analysis of the RT-DETR Defect Detector

October 2026 · model: RT-DETR with 30 object queries (best configuration, 0.786 mAP@0.5) · no retraining

---

## What was done

**Grad-CAM** produces a heatmap of the image regions that most influenced a model's prediction. It was used to see **where the trained model looks** when it detects, or fails to detect, a PCB defect.

- Each heatmap explains **one detection** (one box, one class), not the whole image.
- It was applied to **all 107 defects in the validation set**.
- Each defect was sorted into one of five outcomes: correct, wrong class, bad box, or one of two kinds of miss (see table).

![Grad-CAM example](<gradcam_analysis/correct/categories__Component Liftup__Connecter Liftup (3)_gt0.png>)
*Example: a correctly detected lifted connector pin. Left to right: image, true box (green) and predicted box (red), heatmap, overlay. The heatmap is concentrated on the lifted pin.*

---

## Results

| Outcome | Defects | Share |
|---|:---:|:---:|
| Detected correctly | 83 | 78% |
| **Missed: model boxed the defect but gave it a low score** | **13** | **12%** |
| Missed: model never looked at the defect | 6 | 6% |
| Wrong class | 3 | 3% |
| Box in the wrong place | 2 | 2% |

### Main findings

1. **When the model is right, it usually looks at the right thing.** For 64% of correct detections, the strongest point of the heatmap is on the defect. On Component Liftup, for example, it focuses on the lifted pin rather than the whole component.

2. **The main problem is missed defects (19 of 24 errors), not confusion between classes.** Most misses (13) are *found but not confident*: the model draws a box in the right place but gives it only a 1–10% score. Lowering the confidence threshold would not recover them, because their scores are too low.

3. **Two classes cause most of the misses.**

   | Class | Detected | Missed |
   |---|:---:|:---:|
   | Solder Short | 20 of 30 | 8, mostly "found but not confident" |
   | Component No Solder | 8 of 17 | 7, mostly "never looked" |

   Polarity Wrong is 4 of 7, but has very few examples. Every other class is detected 88–100% of the time.

4. **Wrong-class errors are rare (3).** In one case (No Solder labelled as Solder Dry) the heatmap is exactly on the joint: the model looked at the right place but chose the wrong label.

5. **Defect size is not the main cause.** Missed defects are only slightly smaller than detected ones: a median of 3.2% of the image versus 4.3%.

---

## Suggested next steps

1. **Review the labels for Component No Solder and Solder Dry.** No Solder is the weakest class and the main source of confusion; inconsistent labelling would explain both.
2. **Collect more training examples of Solder Short and No Solder.** The model already finds them but isn't confident.
3. **Check for near-duplicate photos split between the training and validation sets.** Some validation images look like repeat shots of the same part, which would make scores look better than they are.

---

## Limitations

- Grad-CAM shows which regions *influenced* a prediction, not *why* the model decided.
- The heatmap is taken from one internal layer (the middle of three feature scales). Very small details may not appear in it.
- The validation set has only 107 defects, so per-class numbers are small (4–30 per class).

---

## Files and reproducibility

| Path | Contents |
|---|---|
| `gradcam_analysis/full_val/summary.txt` | All counts above |
| `gradcam_analysis/full_val/full_val.csv` | One row per defect: outcome, scores, heatmap location |
| `gradcam_analysis/full_val/<outcome>/` | Heatmap figures for all 24 errors |
| `gradcam_analysis/correct/`, `…/classification_error/`, etc. | The earlier hand-picked pilot examples |
| `gradcam_analysis/gradcam_rtdetr.py` | Code |

<details>
<summary>Technical notes (method checks, how to re-run)</summary>

- **Target:** the final-decoder pre-sigmoid logit of the reported (query, class) pair. For misses, the true class's logit at the query whose box overlaps the defect most. "Found but not confident" means that query's IoU ≥ 0.5.
- **Outcomes** at confidence 0.25: correct = same class, IoU ≥ 0.5; wrong class = other class, IoU ≥ 0.5; bad box = same class, 0.1 ≤ IoU < 0.5; otherwise missed.
- **Layer:** 24 (P4, stride 16), chosen on 10 held-out correct detections by "does the heatmap peak land in the true box". Scores: 8/10, against 3/10 for P3 and 2/10 for P5.
- **Border fix:** the outermost ring of feature cells is zeroed to remove zero-padding artifacts.
- **Checks:** the forward pass matches Ultralytics `predict()` on 15/15 images; gradients are non-zero and finite; heatmaps differ across images (mean correlation 0.05).
- **Re-run:** `sbatch --partition=a40 --qos=a40 --export=ALL,GC_ARGS="scan;full" gradcam_analysis/run_gradcam.sbatch` (about 6 minutes).
- The earlier, longer version of this report, including the 640 vs 1280 px comparison, is kept in `GRADCAM_ANALYSIS_DETAILED.md`.

</details>
