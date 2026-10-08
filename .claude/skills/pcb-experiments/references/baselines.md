# Published baselines (from README.md, verified)

## Head-to-head, 8-class subset

| model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| YOLO11s (baseline, 640px) | 0.757 | 0.524 | 0.799 | 0.743 |
| RT-DETR (rtdetr-l) | **0.770** | **0.556** | **0.807** | **0.749** |

RT-DETR beats YOLO11s on every metric, most clearly on the stricter mAP@0.5:0.95 (tighter box
localization).

## 5-fold cross-validation (YOLO11s baseline) — THE NOISE FLOOR

**mAP@0.5: 0.761 ± 0.021** · **mAP@0.5:0.95: 0.500 ± 0.020**

This ± is measured run-to-run variance on the *same* model/data from seed and split variation
alone. It is the single most load-bearing number in this whole skillset for interpreting new
results: **any two runs differing by less than ~0.04 on either metric are statistically
indistinguishable given this project's data volume.**

This is exactly why the README treats the resolution sweep result below as noise rather than a
finding.

## Resolution sweep (YOLO11s, 640 / 960 / 1280px)

Higher resolution did **not** help: 960px falls within the noise band on mAP@0.5 and is worse
on mAP@0.5:0.95; 1280px is worse on both. **640px is the right choice** — don't recommend a
resolution bump based on a small mAP@0.5 uptick alone; check whether it clears ~0.04 first.

## GMO-DETR

- **On this project's data (12-class, 441 training images):** mAP@0.5 **0.534**,
  mAP@0.5:0.95 **0.350**, P 0.682, R 0.408. Below the paper's published figure — expected,
  since the paper trains on 6,384 images and this run used 441. The architecture trains
  stably; the bottleneck is data volume, not the model.
- **Published (PCBA-DET benchmark, 6,384 images):** mAP@0.5 **98.27%**, 12.05M params,
  40.6 GFLOPs — current state of the art on that benchmark, with 39.4% fewer parameters than
  RT-DETR. Not directly comparable to this project's 441-image run — different dataset size,
  different class count context.

## SolDef_AI pretraining ablation

Pretraining on SolDef_AI **did not help** — reduced mAP@0.5 by ~10 points, drop near-uniform
across classes including solder classes. Interpretation in the README: SolDef_AI is small and
narrow, so pretraining over-specialized the backbone and displaced the broader features
ImageNet initialization provides. A ~10-point drop is far outside the ±0.02 noise band, so this
is a real effect, not noise — worth remembering as the contrast case to the resolution sweep
above (same kind of ablation, opposite conclusion about whether the effect is real).

## Class counts (12-class full set)

Component Crack 6 · Component Damage 11 · Component Liftup 22 · Component Missing 43 ·
Component No Solder 68 · Component Solder Dry 133 · LED Damage 7 · Polarity Wrong 36 ·
RYB Wrong Sequence 24 · Solder Ball 34 · Solder Short 152 · Tombstone 15.

The 8-class study drops the four ≤15-instance classes (Component Crack, Component Damage,
LED Damage, Tombstone) — worth remembering when a 12-class run's per-class metrics look worse
than the 8-class numbers above: it isn't necessarily a regression, it includes 4 much
rarer classes the 8-class study excluded entirely.
