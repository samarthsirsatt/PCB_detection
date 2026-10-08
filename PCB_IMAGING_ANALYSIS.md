# PCB Imaging Analysis

> This document is the current authoritative imaging reference for the PCB project. Values marked
> Measured come from the actual dataset/metadata; Calculated values are derived from those
> measurements; Assumptions are explicitly identified.

**How to read this file.** Every image in our dataset stores its own field of view (FOV) in its
EXIF metadata, so for the first time we can state defect sizes in **millimetres**, not just pixels.
Four labels are used throughout:

| Label | Meaning |
|---|---|
| **Measured** | Read directly out of the image files or the COCO JSON |
| **Calculated** | Arithmetic on measured values |
| **Derived** | Engineering conclusion from measured/calculated values |
| **Assumption** | Not verified by any file — flagged as such |

Raw outputs are in [`imaging/`](imaging/) (see [§13](#13-reproducibility)).

---

## 1. Dataset Facts

| Fact | Value | Status |
|---|---:|---|
| Image files inspected | 1,746 | Measured |
| Defect (annotated) images | 551 | Measured |
| Defect-free ("OK") images | 1,195 | Measured |
| Unique image dimensions | **1280 × 720 px only** (1746/1746) | Measured |
| Aspect ratio | 16:9 (1.7778) for every file | Measured |
| Imaging configurations | 1 sensor format, **27 magnification settings** | Measured |
| COCO annotations | 604 boxes over 547 images, 12 classes | Measured |
| Boxes per image | 1 (513 img), 2 (20), 3 (9), 4 (1), 5 (4) | Measured |
| Images with no box | 4 of 551 | Measured |
| Segmentation masks | none (`segmentation: []` everywhere) | Measured |

There is exactly **one image geometry** but **many magnifications** — this distinction drives most
of the conclusions below.

---

## 2. Camera / Metadata

Every file carries one EXIF tag, `UserComment`, holding a JSON blob:

```
{"fov": 25.0, "unit": 1}
```

| Item | Finding | Status |
|---|---|---|
| `fov` field present | **1746 / 1746 images (100%)** | Measured |
| Meaning of `fov` | horizontal FOV, width of frame | Measured (see check below) |
| Meaning of `unit: 1` | millimetres | Measured (see check below) |
| Camera make / model | **absent** | Measured (not determinable) |
| Lens / objective / NA | **absent** | Measured (not determinable) |
| Working distance | **absent** | Measured (not determinable) |
| Exposure / ISO / f-number / white balance | **absent** | Measured (not determinable) |
| Capture date | absent from EXIF; 1,189 filenames carry a `YYYYMMDDhhmmss` stamp | Measured |
| PCB / panel size | **absent** | Not determinable from current data |

**Unit verification (why we trust `fov`).** 1,189 filenames independently encode the FOV, e.g.
`20260127081757.035_h14v7.875mm.jpg`. For all 1,189: the filename's `h` value equals the EXIF
`fov` exactly, and the filename's `v` value equals `h × 9/16` to within 0.002 mm. Two independent
records agree and the vertical value matches the 16:9 sensor, so `fov` is the **horizontal FOV in
mm** and pixels are **square** (identical px/mm in x and y). This is a cross-checked measurement,
not an assumption.

**Vertical FOV is calculated, never stored:** `FOV_h = FOV_w × 720/1280 = FOV_w × 0.5625`.

---

## 3. Defect Size

Per-class annotated bounding-box statistics over all 604 boxes. `min side` = the shorter of box
width and height — the dimension that actually limits whether a defect is resolvable.
Full min/max/mean/median/sd/p25/p75 for every field are in `imaging/per_class_stats.json`.

| Class | n | median w (px) | median h (px) | median min-side (px) | smallest min-side (px) | median area (px²) | median area (% of frame) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Solder Ball | 43 | 78 | 78 | **72** | **22.8** | 5,605 | 0.61 |
| Component Crack | 8 | 82 | 128 | 82 | 56.2 | 10,003 | 1.09 |
| Component No Solder | 114 | 127 | 140 | 114 | 32.9 | 17,066 | 1.85 |
| Solder Short | 144 | 249 | 130 | 128 | 34.0 | 32,596 | 3.54 |
| Component Damage | 9 | 240 | 160 | 160 | 78.1 | 35,159 | 3.81 |
| Component Liftup | 22 | 534 | 170 | 170 | 99.9 | 97,532 | 10.58 |
| Component Solder Dry | 138 | 188 | 210 | 177 | 54.2 | 42,777 | 4.64 |
| Component Missing | 44 | 393 | 232 | 220 | 98.3 | 90,969 | 9.87 |
| LED Damage | 7 | 265 | 240 | 237 | 195.3 | 63,462 | 6.89 |
| Tombstone | 15 | 431 | 304 | 297 | 228.1 | 134,376 | 14.58 |
| RYB Wrong Sequence | 24 | 529 | 338 | 338 | 196.7 | 171,270 | 18.58 |
| Polarity Wrong | 36 | 509 | 503 | 450 | 250.9 | 279,923 | 30.37 |

*These are **annotated bounding-box dimensions** — the region a human drew in CVAT, not a measured
physical boundary of the defect. Treat them as an upper bound on true defect extent.*

**By the COCO size convention (at native 1280×720):** 530 boxes large (>96²), 72 medium, **2 small
(<32²)**. Both "small" boxes are Solder Ball. By this standard the dataset is **not** a small-object
dataset at native resolution — see §5 for why Solder Ball is nonetheless the hard class.

---

## 4. Pixels/mm

```
px_per_mm_x = image_width_px  / FOV_w_mm = 1280 / fov
px_per_mm_y = image_height_px / FOV_h_mm =  720 / (fov × 0.5625) = 1280 / fov     (identical)
```

FOV is **variable, not constant** — it is a per-image operator zoom setting.

| Quantity (annotated images, n=604 boxes) | min | p25 | median | p75 | max | Status |
|---|---:|---:|---:|---:|---:|---|
| FOV width (mm) | 1.2 | 13.0 | **18.0** | 25.0 | 40.0 | Measured |
| Pixels per mm | 32.0 | 51.2 | **71.1** | 98.5 | 1066.7 | Calculated |

*(The two rows are inverses: `px/mm = 1280 / FOV`, so the smallest FOV gives the largest px/mm.
Each row is quoted in its own natural order.)*

Most-used settings on annotated images: 20 mm (98 boxes), 25 mm (66), 30 mm (61), 18 mm (59),
35 mm (44). Distinct settings: **27** on defect images, 26 on OK images.

**The spread is 33× (32 → 1067 px/mm).** A 1 mm feature is 32 px wide in one image and 1,067 px in
another. This is the single most consequential imaging fact in the dataset.

---

## 5. Smallest / Most Difficult Defects

Two rankings, and they disagree — which is the point.

**By physical size** (p10 of min-side in mm — the small end of each class, more robust than the min):

| Rank | Class | p10 min-side (mm) | median min-side (mm) |
|---:|---|---:|---:|
| 1 | LED Damage | 0.259 | 1.911 |
| 2 | Component Crack | 0.429 | 0.494 |
| 3 | **Solder Ball** | 0.449 | **0.768** |
| 4 | Component No Solder | 0.491 | 2.093 |
| 5 | Component Damage | 0.549 | 0.689 |
| 6 | Solder Short | 0.613 | 2.060 |

Absolute smallest annotated box in the dataset: **0.0434 mm** (Solder Ball, 37 px at 853 px/mm).
Smallest by pixels: **22.8 px** (Solder Ball, 0.446 mm at 51.2 px/mm).

**By pixel size** (§3): Solder Ball → Component Crack → Component No Solder → Solder Short.

**Why the two rankings differ — the central finding.**

| Correlation (log–log, n=604) | Value | Reading |
|---|---:|---|
| bbox area (px²) vs bbox area (mm²) | 0.750 | box pixel size partly tracks physical size |
| FOV vs bbox area (mm²) | 0.686 | operator zooms **out** for physically larger defects |
| **FOV vs bbox area (px²)** | **0.034** | **pixel size is independent of magnification** |

The last row is effectively zero. The operator zoomed until each defect filled a similar fraction
of the frame, whichever defect it was. **Magnification was chosen to normalise apparent size**, so
pixel size in this dataset reflects a human framing decision rather than the physical size of the
defect. Consequences:

1. The model cannot learn any physical size prior — the same defect appears at wildly different
   scales, and different-sized defects appear at similar scales.
2. Solder Ball is the hardest small class **not** because it is under-sampled in pixels (median
   72 px) but because it is genuinely the smallest *physical* defect that is also frequent, and it
   is imaged across a 3.3× px/mm spread.
3. Any px/mm figure quoted for this dataset must be given as a range, never a single number.

---

## 6. Imaging Requirements

`required_px_per_mm = desired_pixels_across_defect / defect_size_mm`, applied to the **p10 min-side
in mm** of each class (i.e. sized to catch the small 10% of each class, not just the median).
Measured-based — no assumed defect size anywhere. Full table: `imaging/required_px_per_mm.csv`.

| Class | p10 min-side (mm) | 3 px | 5 px | 10 px | 15 px | 20 px |
|---|---:|---:|---:|---:|---:|---:|
| LED Damage | 0.259 | 12 | 19 | 39 | 58 | 77 |
| Component Crack | 0.429 | 7 | 12 | 23 | 35 | 47 |
| **Solder Ball** | 0.449 | 7 | 11 | **22** | 33 | **45** |
| Component No Solder | 0.491 | 6 | 10 | 20 | 31 | 41 |
| Component Damage | 0.549 | 5 | 9 | 18 | 27 | 36 |
| Solder Short | 0.613 | 5 | 8 | 16 | 25 | 33 |
| Component Solder Dry | 1.358 | 2 | 4 | 7 | 11 | 15 |
| Component Missing | 1.710 | 2 | 3 | 6 | 9 | 12 |
| Tombstone | 1.960 | 2 | 3 | 5 | 8 | 10 |
| Component Liftup | 2.294 | 1 | 2 | 4 | 7 | 9 |
| RYB Wrong Sequence | 4.223 | 1 | 1 | 2 | 4 | 5 |
| Polarity Wrong | 5.056 | 1 | 1 | 2 | 3 | 4 |

**Read this against §4** (our measured range is **32–1067 px/mm, median 71**):

- **At our median 71 px/mm**, every class clears the 20 px bar except LED Damage, which needs 77 —
  a 9% shortfall on the small 10% of a 7-instance class. The smallest Solder Ball p10 defect gets
  **32 px** across, three times a comfortable 10 px budget.
- **At our coarsest setting, 32 px/mm** (widest 40 mm FOV), six classes drop below 20 px across,
  but every class still clears 10 px except LED Damage (needs 39).
- **At our finest settings** (hundreds of px/mm) we are oversampling by an order of magnitude.

> **Derived:** camera sampling is **not** the limiting factor at native resolution. Every scenario
> in this table is met, or nearly met, by magnifications the rig already uses. The problem is that
> we use all 27 of them, not that any one of them is too coarse.

**FOV implied by sensor width** — `FOV_mm = width_px / px_per_mm` (Calculated):

| px/mm | 1280 px | 1920 px | 2448 px | 4096 px | 5472 px |
|---:|---:|---:|---:|---:|---:|
| 32 (our coarsest) | 40.0 | 60.0 | 76.5 | 128.0 | 171.0 |
| 71.1 (our median) | **18.0** | 27.0 | 34.4 | 57.6 | 77.0 |
| 100 | 12.8 | 19.2 | 24.5 | 41.0 | 54.7 |
| 200 | 6.4 | 9.6 | 12.2 | 20.5 | 27.4 |

**Model input resolution is a separate axis.** At the 640 px training size our 1280 px frames are
halved, so the median Solder Ball min-side drops **72 px → 36 px** and the smallest observed drops
**22.8 px → 11.4 px**. Pixels are being discarded in the training pipeline, not lost at the camera.

---

## 7. Full Board vs ROI / Tiled

| Option | What the data says | Verdict |
|---|---|---|
| **A. Full-board single image** | Largest FOV in the entire dataset is **40 mm**. Board size is **unknown** — never recorded. No image in the dataset shows a whole board. | **Not supported by any file.** Cannot be evaluated. |
| **B. Tiled board** | Would need board dimensions and a stage/positioning record. Neither exists in the metadata. | Not evaluable from current data. |
| **C. Component / ROI imaging** | **This is what the dataset already is** — 1746/1746 images are 1.2–40 mm close-ups of individual components. | **Already our setup; keep it.** |

**Derived conclusion:** our dataset is a component/ROI dataset and the research should stay there.
Full-board is not a decision we can currently make — a 150 mm board at our median 71 px/mm would
need ~114 MP (calculated), but *150 mm is an assumption, not a measured board size*, so that number
is illustrative only and must not be quoted as a requirement.

---

## 8. Relation to Current Model Results

Per-class scores from the best measured configuration, **RT-DETR with 30 object queries**
(`runs/experiments/nq30`), which is an **8-class** benchmark. The four rarest classes (Component
Crack 8, Component Damage 9, LED Damage 7, Tombstone 15 boxes) are excluded there but remain part
of the 12-class project scope; their rows are left blank rather than mixed in.

| Class | min-side (px, median) | at 640 input | min-side (mm, median) | px/mm spread (p90/p10) | AP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Solder Ball | 72 | 36 | 0.77 | 3.3× | 0.783 | **0.422** |
| Component No Solder | 114 | 57 | 2.09 | 2.7× | **0.472** | **0.365** |
| Solder Short | 128 | 64 | 2.06 | 3.0× | 0.719 | 0.452 |
| Component Liftup | 170 | 85 | 3.01 | 1.8× | 0.895 | 0.733 |
| Component Solder Dry | 177 | 88 | 1.87 | 2.9× | 0.839 | 0.612 |
| Component Missing | 220 | 110 | 3.63 | 2.5× | 0.984 | 0.710 |
| RYB Wrong Sequence | 338 | 169 | 7.80 | 2.5× | 0.995 | 0.661 |
| Polarity Wrong | 450 | 225 | 9.22 | 1.8× | 0.627 | 0.581 |
| Component Crack | 82 | 41 | 0.49 | 2.1× | — | — |
| Component Damage | 160 | 80 | 0.69 | 1.8× | — | — |
| LED Damage | 237 | 118 | 1.91 | 10.0× | — | — |
| Tombstone | 297 | 148 | 2.66 | 1.5× | — | — |

Rank correlations across the 8 scored classes (Spearman):

| Pair | ρ vs AP@0.5 | ρ vs mAP@0.5:0.95 |
|---|---:|---:|
| median min-side in **px** | +0.38 | **+0.57** |
| median min-side in **mm** | +0.21 | +0.45 |
| **px/mm spread** (scale inconsistency) | −0.26 | **−0.69** |

**Observations (n = 8 classes; none of these reaches statistical significance, which needs
\|ρ\| > 0.71 — treat as suggestive only):**

- Classes with **more pixels across them** tend to be localised better, and the effect is stronger
  on the strict mAP@0.5:0.95 (+0.57) than on AP@0.5 (+0.38). This is **consistent with** localisation,
  not classification, being scale-sensitive — matching the measured 0.907 classification ceiling vs
  0.743 detection score already on record.
- **Scale inconsistency tracks poor localisation more strongly than size does** (ρ = −0.69 against
  mAP@0.5:0.95). Suggests the variable-magnification problem of §5 may matter more than absolute
  smallness. Not established — and **subsequently tested and not supported**: a per-instance
  re-evaluation of the same model over all 107 validation defects found ρ = −0.028 for px/mm vs
  box IoU, with between-bin differences no larger than chance (p = 0.69). See
  [`FOV_LOCALIZATION_ANALYSIS.md`](FOV_LOCALIZATION_ANALYSIS.md). Defect **pixel size** does hold up
  (ρ = +0.294, p ≈ 0.002); variable FOV itself does not.
- **Some large defects are still difficult**: Polarity Wrong is the physically largest class
  (9.2 mm median) yet scores only 0.627 AP@0.5. Its failure mode is semantic (orientation), not
  scale — size explains part of the picture, not all of it.
- **Component No Solder is the weakest class (0.472)** and is mid-sized. Size does not explain it;
  the existing annotation-consistency hypothesis vs Component Solder Dry remains the better lead.

---

## 9. Already Tested

Imaging-related work already completed — **do not repeat these**:

| Experiment | Where | Outcome |
|---|---|---|
| Model input resolution 640 / 960 / 1280, batch-confounded | README (8-class) | inconclusive — two variables changed at once |
| Model input resolution re-run, batch fixed at 8 | OPTIMIZATION_STUDY Part 1 | **1280 px = 0.787, +0.044 over 640** — resolution helps; adequately settled |
| Rectangular training (removes ~44% letterbox padding at 640) | OPTIMIZATION_STUDY | +0.001, no effect |
| Stride-4 (P2) small-object head, GMO-DETR | GMO_STUDY Part 3 | mean unchanged; raised the Solder Ball floor (never below 0.198 vs 0.105/0.109) |
| Stride-4 (P2) head, **RT-DETR** | — | **never run** — the one genuine gap |
| Object queries 300 → 30 | OPTIMIZATION_STUDY | +0.046, confirmed twice; best result on the dataset |
| Localisation vs classification split | OPTIMIZATION_STUDY | 0.907 classification given a perfect box vs 0.743 detection — error is localisation |
| Per-image FOV / px/mm / physical defect size | **this document** | **new — first time measured** |

Keep the axes distinct: **model input resolution ≠ camera resolution ≠ optical resolution ≠ FOV ≠
px/mm.** Everything before this document was model input resolution. This document is the first
measurement of the other four.

---

## 10. Recommended Next Experiments

All five use the existing dataset. No new hardware, no new data collection.

1. ~~**Scale-normalised training.**~~ **Deprioritised.** The ρ = −0.69 signal it targeted did not
   survive the per-instance test in [`FOV_LOCALIZATION_ANALYSIS.md`](FOV_LOCALIZATION_ANALYSIS.md)
   (ρ = −0.028, between-bin differences at chance level). Worth revisiting only if a larger
   validation set restores the signal.
2. **Stride-4 (P2) head on RT-DETR.** Recommended by both prior studies and still untested there.
   RT-DETR's low variance (±0.021 vs GMO's ±0.069) is what makes the answer readable at all.
3. ~~**Slice the existing results by px/mm.**~~ **Done** —
   [`FOV_LOCALIZATION_ANALYSIS.md`](FOV_LOCALIZATION_ANALYSIS.md).
4. **Fix the metadata pipeline going forward.** Record camera model, lens, working distance and
   board ID at capture time. All four are missing today and each is one line of EXIF. Cheap now,
   impossible retroactively.
5. **Only if 1–3 leave a gap:** a small controlled capture of ~20 Solder Ball defects at three
   fixed magnifications, to separate "small defect" from "inconsistent magnification" with real
   images rather than resampling.

**Explicitly not recommended:** raising camera resolution or px/mm (§6 shows sampling is already
sufficient), full-board imaging (§7 — board size unknown), a new dataset, or any architecture
redesign.

---

## 11. Unknowns / Assumptions

| Unknown | Status |
|---|---|
| PCB / panel physical size | Not determinable from current data — no file records it |
| Camera make, model, sensor size, pixel pitch | Not determinable from current data |
| Lens, magnification optics, numerical aperture | Not determinable from current data |
| Working distance, depth of field | Not determinable from current data |
| True **optical** resolution (MTF / blur) | Not determinable — px/mm is *sampling*, and sampling ≥ optical resolution is not guaranteed. All §6 conclusions are about sampling only |
| Lighting configuration, illumination uniformity | Not recorded in any file |
| Whether all 1,746 images come from one physical rig | **Assumption** — identical 1280×720 geometry and one shared metadata schema make it likely, but nothing verifies it |
| Real physical defect extent | **Assumption if used** — we measured annotated boxes, which bound the defect from outside |
| 150 × 150 mm board (used in earlier discussions) | **Assumption — not supported by any file. Do not quote it as a fact.** |

---

## 12. Final Reference Table

| Quantity | Value | Unit | Source | Status |
|---|---:|---|---|---|
| Image resolution | 1280 × 720 (1746/1746) | px | actual files | Measured |
| Aspect ratio | 1.7778 (16:9) | — | actual files | Measured |
| FOV width, range | 1.2 – 40.0 | mm | EXIF `UserComment` | Measured |
| FOV width, median (annotated) | 18.0 | mm | EXIF `UserComment` | Measured |
| FOV height | FOV_w × 0.5625 | mm | 720/1280 | Calculated |
| Pixels/mm, range | 32.0 – 1066.7 | px/mm | 1280 / FOV | Calculated |
| Pixels/mm, median (annotated) | 71.1 | px/mm | 1280 / FOV | Calculated |
| Distinct magnification settings | 27 | — | EXIF | Measured |
| Smallest bbox (min side) | 22.8 | px | COCO | Measured |
| Smallest physical bbox (min side) | 0.0434 | mm | COCO + FOV | Calculated |
| Smallest frequent class, median size | 0.768 (Solder Ball) | mm | COCO + FOV | Calculated |
| Required px/mm, 10 px on Solder Ball p10 | 22 | px/mm | 10 / 0.449 | Derived |
| Required px/mm, 20 px on Solder Ball p10 | 45 | px/mm | 20 / 0.449 | Derived |
| Required px/mm, 20 px on worst class (LED Damage) | 77 | px/mm | 20 / 0.259 | Derived |
| Practical FOV (current, keep) | 13 – 25 | mm | measured IQR of annotated images | Derived |
| Sampling adequacy at native resolution | sufficient for all 12 classes | — | §6 vs §4 | Derived |
| PCB size | unknown | mm | — | Not determinable |
| Camera / lens / working distance | unknown | — | — | Not determinable |

---

## 13. Reproducibility

| File | Contents |
|---|---|
| `imaging/audit_images.py` | walks all 1,746 files, dumps geometry + every EXIF tag |
| `imaging/analyze.py` | EXIF FOV extraction, filename cross-check, COCO join, per-class stats |
| `imaging/extra.py` | size buckets, correlations, smallest-annotation listings |
| `imaging/requirements.py` | required px/mm and FOV-vs-sensor tables |
| `imaging/link.py` | defect size ↔ `nq30` per-class AP join and Spearman ρ |
| `imaging/image_audit.csv` | 1,746 rows — geometry, EXIF keys, raw UserComment |
| `imaging/image_fov.csv` | 1,746 rows — EXIF FOV vs filename FOV cross-check |
| `imaging/annotations_measured.csv` | **604 rows — every box in px, ratios, px/mm, mm, mm²** |
| `imaging/per_class_stats.{csv,json}` | full min/p25/median/mean/p75/max/sd per class per field |
| `imaging/required_px_per_mm.csv`, `fov_vs_resolution.csv`, `size_vs_performance.csv` | §6 and §8 tables |

Run with `envs/pcb/bin/python`; requires only Pillow.

---

## Bottom Line

1. Our images are **1280 × 720, every one of them**, and each carries its own FOV in EXIF — so
   physical scale is recoverable for the whole dataset. This was not known before.
2. Actual FOV is **1.2–40 mm (median 18 mm)** and actual sampling is **32–1067 px/mm (median 71)**.
   Both are per-image operator zoom settings, not fixed rig properties.
3. The smallest annotated defect is **0.043 mm**; the smallest frequent class, Solder Ball, has a
   median short side of **0.77 mm / 72 px**.
4. Putting 20 pixels across the smallest 10% of every class needs at most **77 px/mm**; our median
   is 71 and even our coarsest setting (32) clears 10 px for 11 of 12 classes. **Camera sampling is
   not our bottleneck.**
5. The real imaging defect is **scale inconsistency**: a 33× px/mm spread, with magnification chosen
   to normalise how big defects *look* (FOV↔px-area correlation 0.03). The model sees no consistent
   physical scale.
6. That inconsistency *appeared* to track poor localisation at class level (ρ = −0.69, n=8), but a
   per-instance test on all 107 validation defects did **not** confirm it (ρ = −0.028; see
   [`FOV_LOCALIZATION_ANALYSIS.md`](FOV_LOCALIZATION_ANALYSIS.md)). Defect **pixel size** is the
   variable that survives (ρ = +0.294, p ≈ 0.002).
7. Training at 640 px halves every defect (Solder Ball 72 → 36 px). **We throw away resolution in
   the model, we do not lack it in the camera.**
8. Next: normalise scale using the px/mm we can now compute per image, add the stride-4 head to
   RT-DETR, and re-slice existing results by px/mm — all on the current dataset.
9. Full-board imaging cannot be assessed: **no file records the board size**, and the widest view we
   own is 40 mm. Our work is, and should stay, component/ROI imaging.
10. Fix capture metadata now — camera, lens, working distance, board ID are all missing and cannot
    be recovered later.
