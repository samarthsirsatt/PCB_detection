# Grad-CAM Analysis of RT-DETR (30 queries)

October 2026 · post-hoc only — no training, no fine-tuning, no architecture change · checkpoint `runs/experiments/nq30/weights/best.pt`

---

## Summary

Grad-CAM was applied to individual detections of the optimized 30-query RT-DETR, on 11 validation cases chosen to cover every error type, plus a 10-image held-out set used only to choose the layer.

- **The method works on this model.** Each heatmap explains one specific detection: one decoder query's logit for one class. Its gradients are exact, non-zero and finite. On held-out correct detections, the heatmap peak falls inside the ground-truth box 8 times in 10.
- **On correct detections the model looks at the defect itself, not the component.** On all three Component Liftup cases, the activation sits on the lifted lead.
- **Most errors are missed detections, not mislabelled ones.** At confidence 0.25 the 107 validation defects split into 83 correct, 19 missed, 3 wrong class and 2 poor box. In two of the three misses inspected, a query was already sitting on the defect but scored it near zero. The model found the defect but didn't commit to it.
- **One clear confusion:** a Component No Solder joint predicted as Component Solder Dry, with the activation exactly on the joint. Right place, wrong label. Two of the three validation classification errors are predicted Solder Dry.

Two implementation problems had to be fixed before any of this could be trusted (Part 2). With either one left in, the conclusions would have been wrong.

---



## Part 1 — Method

**What is being explained.** RT-DETR has no per-pixel class map. Its output is 30 decoder queries, each with a box and 8 class logits. Ultralytics reports a detection for each (query, class) pair whose sigmoid score exceeds the confidence threshold. The Grad-CAM target is the **pre-sigmoid logit** `dec_scores[query, class]` **of the final decoder layer** for the reported pair. Its gradient therefore explains that one detection and nothing else.

For a **missed** defect there is no detection to explain. The target is then the ground-truth class's logit at the query whose box overlaps the ground truth most, whatever its score. The question this answers is: what does the nearest query see when asked about the true class?

**Exactness check.** The script runs its own forward pass to keep the gradient graph. On 15 validation images it reproduced Ultralytics' `predict()` exactly in class and box (IoU > 0.99). Confidence agreed within 0.002 on all of them, and within 0.001 on 13. The remaining difference is floating-point noise, confirmed by an unfused fresh model giving the same reference values.

**Grad-CAM.** Channel weights are the spatially averaged gradients, and the map is ReLU(Σ wₖAₖ). It is upsampled to the network input and stretched back to the original 1280×720 frame. RT-DETR stretches its input rather than letterboxing, so this is the exact inverse.

**Error categories** (per ground-truth box, detections at confidence ≥ 0.25):


| Category             | Rule                                                         |
| -------------------- | ------------------------------------------------------------ |
| Correct              | Same-class detection with IoU ≥ 0.5                          |
| Classification error | No correct match, but a wrong-class detection with IoU ≥ 0.5 |
| Localization error   | Best same-class detection has 0.1 ≤ IoU < 0.5                |
| Missed detection     | None of the above                                            |


---



## Part 2 — Choosing the layer, and two problems found

Three candidate layers were tested. They are the three feature maps the decoder reads, all with 256 channels:


| Layer             | Scale         | Map at 640 px |
| ----------------- | ------------- | ------------- |
| 21 `fpn_blocks.1` | P3, stride 8  | 80×80         |
| 24 `pan_blocks.0` | P4, stride 16 | 40×40         |
| 27 `pan_blocks.1` | P5, stride 32 | 20×20         |


**Problem 1 — the first selection metric picked the wrong layer.** Ranking layers by "share of heatmap energy inside the GT box, relative to the box's area" scored all three about equally (1.62 / 1.56 / 1.48) and chose layer 21. Visual inspection showed layer 21 is speckle noise spread over the whole frame, a known Grad-CAM failure at high resolution. The metric rewards uniform noise whenever the GT box is large. It was replaced with the **pointing game** (does the heatmap's peak land inside the GT box?). The new metric is scored on 10 correct detections kept separate from the figure set, so the layer isn't chosen on the same images it is then used to interpret.

**Problem 2 — border artifacts.** Many heatmaps had their maximum on the outermost row or column of the image, unrelated to any defect. This comes from zero-padding in the convolutions. It hijacked both the pointing game and the colour scale, which washed out the real signal (Figure: `small_objects/…C12…`). The outermost ring of feature cells is now zeroed before upsampling. The share of energy that sat there is recorded for every figure (`border_energy_raw`, 1–50%, median 20%). A faint edge band one cell further in survives on a few images (e.g. J10); it doesn't affect any conclusion below.

**Result on the 10 held-out correct detections:**


| Layer       | Pointing game, before border fix | Pointing game, after | Energy in GT ÷ GT area |
| ----------- | -------------------------------- | -------------------- | ---------------------- |
| 21 (P3)     | 1/10                             | 3/10                 | 1.2×                   |
| **24 (P4)** | 4/10                             | **8/10**             | **4.4×**               |
| 27 (P5)     | 1/10                             | 2/10                 | 1.6×                   |


**Layer 24 was selected** and is used in every figure below. Per-case layer strips are in `gradcam_analysis/layer_comparison/`.

### Sanity checks


| Check                                   | Result                                                                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Gradients non-zero                      | Mean |∂logit/∂A| between 2.6×10⁻⁶ and 1.1×10⁻⁴ on every case                                                              |
| No NaN / Inf                            | All activations, gradients and maps finite                                                                                |
| Target really is the detection          | Forward pass matches `predict()` on 15/15 images (above)                                                                  |
| Maps differ across targets, same image  | Correlation with a different query's map: −0.26 to 0.67, mean 0.27. The higher values are same-class neighbouring queries |
| Maps differ across images               | Mean pairwise correlation 0.05, max 0.47                                                                                  |
| Layers give sensible, different results | Yes — see table above                                                                                                     |


---



## Part 3 — Error analysis

Figures are in `gradcam_analysis/<category>/`, one 4-panel figure per row: original; ground truth (green) and prediction (red) with class, confidence, IoU and query; heatmap; overlay. "CAM focus" comes from reading each figure, cross-checked against the recorded metrics.


| Image                       | True class          | Predicted class      | Conf   | IoU  | CAM focus                                                                                      | Interpretation                                                                                 |
| --------------------------- | ------------------- | -------------------- | ------ | ---- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Component Liftup            | Component Liftup    | Component Liftup     | 0.94   | 0.96 | On the lifted right-hand lead, plus the component's left edge                                  | Correct, for the right reason: the evidence is the lifted lead                                 |
| Liftup (3)                  | Component Liftup    | Component Liftup     | 0.94   | 0.95 | Two bands along both leads, peak at the lifted lead's foot                                     | As above. This appears to be the same part photographed again                                  |
| Connecter Liftup (3)        | Component Liftup    | Component Liftup     | 0.91   | 0.81 | Tight hotspot on the lifted connector pin; 26% of energy in 9% of frame                        | Clearest case: a single pin drives the detection                                               |
| Q2 component lead no solder | Component No Solder | Component No Solder  | 0.36   | 0.37 | Weak, diffuse; faint blob over the left pads, peak outside GT                                  | Low-confidence guess on the adjacent pad. No strong evidence anywhere                          |
| R17 Q2 SOLDER SHORT         | Solder Short        | Solder Short         | 0.84   | 0.25 | Scattered speckle, nothing concentrated on the bridge                                          | Inconclusive. The box covers the whole transistor while GT marks only the bridge               |
| J1 Connector No solder      | Component No Solder | Component Solder Dry | 0.93   | 0.79 | Concentrated on the joint; peak in GT, 10% of energy in 2% of frame                            | **Right place, wrong label.** No Solder vs Solder Dry confusion                                |
| D2 LED polarity wrong       | Polarity Wrong      | Component Solder Dry | 0.90   | 0.86 | Scattered speckle; nothing on the LED or its polarity marking                                  | Uninformative: no localized positive evidence for the wrong class at this layer                |
| BLACK wire no solder        | Component No Solder | (none)               | —      | 0.10 | Diffuse; no focus                                                                              | **No query near the defect** (best overlap 0.10). Missed at query selection                    |
| J10 no solder (1)           | Component No Solder | (none)               | p≈0.01 | 0.21 | Hotspot at the joint region between the pads, just off GT                                      | The model attends near the defect but gives it almost no No Solder score                       |
| C12 solder boll short       | Solder Ball         | Solder Ball          | 0.34   | 0.83 | Compact blob centred on the ball, plus streaks on the neighbouring pins' edges (peak on a pin) | Detected, low confidence. Evidence is the ball *and* its neighbouring leads                    |
| C1 Q1 SOLDER SHORT          | Solder Ball         | (none)               | p≈0.02 | 0.71 | Peak on the solder fillet just below the ball; nothing on the ball                             | **A query sits on the ball** (IoU 0.71) but scores it ~0. Tiny object: 50×53 px, 0.3% of frame |


Selection was deterministic: the first cases alphabetically within each category, not chosen for how they looked. The validation set has only two localization errors and three classification errors in total, so those rows cover nearly all of them.

### Where the errors are, across the whole validation set

The scan (predictions only, no Grad-CAM) over all 107 validation defects:


| Class                | Defects | Correct | Missed | Wrong class | Poor box |
| -------------------- | ------- | ------- | ------ | ----------- | -------- |
| Solder Short         | 30      | 20      | **8**  | 1           | 1        |
| Component Solder Dry | 27      | 26      | 1      | —           | —        |
| Component No Solder  | 17      | 8       | **7**  | 1           | 1        |
| Component Missing    | 9       | 9       | —      | —           | —        |
| Solder Ball          | 8       | 7       | 1      | —           | —        |
| Polarity Wrong       | 7       | 4       | 2      | 1           | —        |
| RYB Wrong Sequence   | 5       | 5       | —      | —           | —        |
| Component Liftup     | 4       | 4       | —      | —           | —        |
| **Total**            | **107** | **83**  | **19** | **3**       | **2**    |


---



## Part 4 — 640 px vs 1280 px

The same cases were run through the 1280 px checkpoint trained with the same 30 queries (`nq30_r1280`). Figures are in `gradcam_analysis/res_640_vs_1280/`.

Note the context: with 30 queries, 1280 px was **worse** on average (0.745 vs 0.786 [mAP@0.5](mailto:mAP@0.5), OPTIMIZATION_STUDY.md Part 6). The study's earlier finding that 1280 px helps applies only to the original 300-query model.


| Case                      | 640 px                       | 1280 px                          | Heatmap                                                |
| ------------------------- | ---------------------------- | -------------------------------- | ------------------------------------------------------ |
| C12 Solder Ball           | Correct, conf 0.34, IoU 0.83 | Correct, **conf 0.86**, IoU 0.91 | Tighter on the ball at 1280; peak moves inside GT      |
| C1 Q1 Solder Ball (50 px) | Missed                       | Missed                           | At neither resolution does activation land on the ball |
| Q2 No Solder              | Poor box, IoU 0.37           | Missed                           | —                                                      |
| R17 Solder Short          | Poor box, IoU 0.25           | Poor box, IoU 0.40               | Still not on the bridge                                |
| J1 No Solder              | Solder Dry, 0.93             | Solder Dry, 0.90                 | Same confusion at both                                 |
| D2 Polarity Wrong         | Solder Dry, 0.90             | Solder Dry, 0.88                 | Same confusion at both                                 |
| BLACK wire, J10           | Missed                       | Missed                           | —                                                      |


On the one detected Solder Ball, higher resolution raised confidence from 0.34 to 0.86 with a more focused heatmap. That fits Solder Ball being the class that gained most from resolution in the study. The 50-pixel ball is missed at both resolutions, and nothing else changes category for the better. This is one image per outcome and supports nothing beyond that.

---



## Part 5 — Answers

1. **Does Grad-CAM work on the existing RT-DETR?** Yes, once two problems were fixed: the layer-selection metric and the border artifacts. The target maps exactly to Ultralytics' detections, gradients are non-zero and finite, maps differ across targets and images, and on held-out correct detections the peak lands on the defect 8 times in 10.
2. **Which layer?** Layer 24 (`pan_blocks.0`, P4, stride 16, 40×40 at 640 px), chosen by pointing game on detections not used for the figures.
3. **What is being explained?** The final-layer pre-sigmoid logit of one (query, class) pair, the exact pair Ultralytics reports. For a missed defect, it's the true class's logit at the nearest query.
4. **Correct detections?** The activation is on the defect, specifically the lifted lead rather than the component body, in all three Component Liftup cases.
5. **Localization errors?** Both cases show weak, scattered maps. This is *not* the "right evidence, misplaced box" pattern; the model had little confident evidence anywhere. There are only two such cases in the whole validation set.
6. **Classification errors?** J1 is a clean "right place, wrong label" (No Solder → Solder Dry). D2 is uninformative. Two of the three validation classification errors are predicted Solder Dry.
7. **Missed and small defects?** Two different failure modes appear.
  - **No query near the defect** (BLACK wire, best overlap 0.10): the defect was never proposed.
  - **A query on the defect but a near-zero score** (C1 Q1, IoU 0.71, p ≈ 0.02; J10, p ≈ 0.01). For the 50-pixel solder ball the activation sits on the adjacent fillet, not the ball.
8. **Do the observations support the numerical findings?** Mostly, with one refinement.
  - *Supports:* errors are about finding defects, not telling them apart. Misses are 19 of the 24 errors at this threshold, consistent with the 0.907 classification ceiling. The No Solder ↔ Solder Dry confusion seen here is the one the study flagged for annotation review (Part 5, step 6). Solder Ball at 1280 px behaves as the resolution result predicts.
  - *Refines:* "finding" failures are not all box-geometry failures. In two of three inspected misses a query already covered the defect and only the confidence was missing. That is a scoring problem, not a localization problem.

---



## Part 6 — Full validation run (all 107 defects)

Same method, layer 24, every validation defect. Output: `gradcam_analysis/full_val/` (`full_val.csv`, `summary.txt`, figures for the 24 errors).


| Outcome                                                          | Count        | Heatmap peak on defect |
| ---------------------------------------------------------------- | ------------ | ---------------------- |
| Correct                                                          | 83 (78%)     | 53/83 (64%)            |
| Missed — a query was on the defect (IoU ≥ 0.5) but scored it low | **13 (12%)** | 3/13                   |
| Missed — no query near the defect                                | 6 (6%)       | 0/6                    |
| Wrong class                                                      | 3 (3%)       | 1/3                    |
| Bad box                                                          | 2 (2%)       | 0/2                    |


**Main issues**

1. **Most misses are "found but not confident" (13 of 19).** The box is right, but the class probability is only 0.01–0.10. Lowering the confidence threshold would not fix this: only 2 of the 13 are above 0.05. 7 of the 13 are Solder Short.
2. **Component No Solder is the weakest class.** Only 8 of 17 are detected. 5 of the 6 "no query near the defect" misses are No Solder, and the heatmap lands on the defect for only 5 of 17.
3. **Class confusion is rare** (3 cases), and two of the three are predicted as Solder Dry.
4. **Missed defects are somewhat smaller** (median 3.2% of the frame vs 4.3% for correct ones), but size alone doesn't explain them.

---



## Limitations

- **Eleven cases is a pilot.** It was meant to verify the method, and individual interpretations are anecdotes. The patterns in Part 5 need the larger run before going into a thesis.
- **One layer shows only part of the evidence.** The decoder also reads P3 (layer 21) directly, and that path bypasses layer 24. Evidence the model takes from P3 alone, which matters most for tiny objects like the 50-pixel ball, won't appear in a layer-24 map. "Nothing on the ball" means nothing *at P4*.
- **Grad-CAM shows correlation with the logit, not reasoning.** All interpretations above are diagnostic, not claims about the model's internals.
- **Possible duplicate images.** "Component Liftup" and "Liftup (3)" look like the same part photographed twice. If near-duplicates also cross the train/val split, validation scores are optimistic. Worth checking separately.

---



## Reproducing

All code is in `gradcam_analysis/gradcam_rtdetr.py`; jobs run through `gradcam_analysis/run_gradcam.sbatch`.

```bash
cd /home/agipml/samarth.sirsat/DDP/pcb
# 1. predict the val split, categorize every GT box, verify against predict()  (~3 min)
sbatch --partition=a40 --qos=a40 --export=ALL,GC_ARGS="scan" gradcam_analysis/run_gradcam.sbatch
# 2. Grad-CAM figures + layer choice, then the 640 vs 1280 comparison           (~2 min)
sbatch --partition=a40 --qos=a40 --export=ALL,GC_ARGS="cam;compare --n 8" gradcam_analysis/run_gradcam.sbatch
```

Useful options: `--layer 21|24|27` (override the layer choice), `--keep-border` (disable border suppression), `--weights` / `--imgsz` (another checkpoint), `--n-select` (size of the held-out layer-selection set).

```
gradcam_analysis/
  gradcam_rtdetr.py            scan / cam / compare
  run_gradcam.sbatch
  scan_nq30.csv                every val GT box: category, prediction, IoU, query
  scan_nq30_predict_check.json forward-pass vs predict() verification
  summary.csv, summary.json    per-figure table + layer scores + sanity checks
  correct/ localization_error/ classification_error/ missed_detection/ small_objects/
  layer_comparison/            same cases at layers 21 / 24 / 27
  res_640_vs_1280/             side-by-side + compare.json
```

