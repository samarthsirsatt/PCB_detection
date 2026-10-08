# GMO-DETR Optimization Study

August 2026 · 18 training runs · Prajna HPC cluster (NVIDIA L40S / A40)
Companion to [`OPTIMIZATION_STUDY.md`](OPTIMIZATION_STUDY.md), which covers RT-DETR.

---

## Summary

GMO-DETR was run on the **same 8-class dataset, the same SEED=42 split, and the same
augmentation** as the RT-DETR optimization study, then tuned with five controlled changes.

The result is negative, and the reason is measurement, not tuning. **GMO-DETR's run-to-run
spread on this dataset is ±0.069 mAP@0.5 — more than three times the ±0.021 noise floor the
RT-DETR study established — and 2 of 18 runs collapsed outright to ~0.13.** Three seeds of the
*identical* baseline configuration scored 0.489, 0.390 and 0.133. Every apparent gain from a
single run in this study is smaller than that spread, so **no GMO-DETR optimization is
established by this data.**

The best-measured configurations improve the baseline from 0.440 to about 0.478 mAP@0.5
(+0.038, within noise). That leaves GMO-DETR roughly **0.31 mAP@0.5 below optimized RT-DETR
(0.786)** — a gap about fifteen times the noise floor, and one that none of these optimizations
comes close to closing.

The cause is not the reimplementation. GMO-DETR is a custom architecture, so **no pretrained
weights exist for it and it trains from scratch on 441 images**, while RT-DETR starts from
COCO-pretrained `rtdetr-l.pt`. The published GMO-DETR result uses 6,384 images. At this dataset
size the initialization advantage dominates every architectural difference tested.

---

## Part 1 — Baseline

`experiments/exp_gmodetr.py` reuses `exp_common.py` — the same COCO→YOLO conversion, the same
rarest-class-stratified split, the same 120 negatives, the same augmentation — so the numbers
below are directly comparable to the RT-DETR table. Training follows the paper's Table 2 as
implemented in `gmo_detr_full_pipeline.py`: 640 px, batch 8, AdamW, lr 1e-4, wd 1e-4, 200
epochs, patience 30.

| | GMO-DETR baseline | RT-DETR baseline |
|---|:---:|:---:|
| mAP@0.5 | **0.440** (mean of 2 valid runs; 0.489 / 0.390) | 0.743 |
| mAP@0.5:0.95 | **0.252** | 0.479 |
| Precision | 0.415 | 0.713 |
| Recall | 0.452 | 0.724 |
| Parameters | **17.9 M** | 32.0 M |
| GFLOPs | **58.0** | 105.4 |
| Inference | 9.3 ms/img | 5.8 ms/img |
| Training | 0.73 h / 188 epochs | 0.32 h / 90 epochs |
| Initialization | **from scratch** | COCO-pretrained |

Per-class AP@0.5 at baseline: RYB Wrong Sequence 0.634 · Component Liftup 0.496 ·
Solder Short 0.536 · Component Solder Dry 0.412 · Component Missing 0.345 · Polarity Wrong
0.330 · Component No Solder 0.262 · **Solder Ball 0.109**.

GMO-DETR is genuinely the lighter model — 44% fewer parameters and 45% fewer FLOPs than
RT-DETR-l, consistent with the paper's efficiency claim. It is nonetheless slower per image
(9.3 ms vs 5.8 ms): the depthwise strip convolutions in CAA and the gated depthwise blocks in
DMambaOut are FLOP-cheap but poorly suited to GPU throughput.

---

## Part 2 — What was tested

| Change | Relevant to GMO-DETR? | Why |
|---|---|---|
| **A. Resolution 640 → 1280** | yes | source images are natively 1280×720 |
| **B. Object queries 300 → 30** | yes | GMO-DETR keeps RT-DETR's `RTDETRDecoder` unchanged; 1.11 objects/image |
| **C. Vertical flip off** | yes | Polarity Wrong and RYB Wrong Sequence are orientation-defined |
| **D. Negatives 120 → 300** | yes | 300 was the measured optimum for RT-DETR |
| **E. Stride-4 head for small objects** | yes | Solder Ball is the one small class (0.61% median area) |

Query count (B) was worth testing precisely because GMO-DETR inherits the RT-DETR decoder
intact — the paper replaces the backbone, neck and AIFI, but not the query mechanism.

The stride-4 head (E) was nearly free to add: `GMONet` already computes a 128-channel stride-4
feature map in its first GMO-Block and discards it. `GMONetBackboneP2` emits it, and the neck
gains one FPN/PAN level — 20.3 M parameters instead of 17.9 M, no redesign.

---

## Part 3 — Results

Every configuration, grouped; runs marked † collapsed (early-stopped below 0.2) and are excluded
from the means.

| Configuration | runs (mAP@0.5) | mean | mAP@0.5:0.95 | vs baseline |
|---|---|:---:|:---:|:---:|
| Baseline | 0.489 / 0.390 / 0.133† | 0.440 | 0.252 | — |
| Vertical flip off | 0.491 / 0.464 | **0.478** | 0.262 | +0.038 *(within noise)* |
| flipud0 + neg300 + P2 | 0.520 / 0.495 / 0.408 | **0.475** | **0.275** | +0.035 *(within noise)* |
| 1280 px | 0.451 | 0.451 | 0.224 | +0.011 *(within noise)* |
| 300 negatives | 0.491 / 0.403 | 0.447 | 0.247 | +0.007 *(within noise)* |
| flipud0 + neg300 | 0.430 | 0.430 | 0.266 | −0.010 *(within noise)* |
| flipud0 + P2 | 0.410 | 0.410 | 0.235 | −0.030 *(within noise)* |
| P2 head (stride-4) | 0.445 / 0.361 | 0.403 | 0.248 | −0.037 *(within noise)* |
| 30 object queries | 0.387 | 0.387 | 0.205 | −0.053 *(within noise)* |
| flipud0 + neg300 + 1280 | 0.528 / 0.128† | 0.528 | 0.301 | +0.088 *(1 valid run)* |

**Baseline seed spread: ±0.069 (0.489 / 0.390, excluding the collapse). Every delta in the
table is smaller than that.**

### The variance is the finding

The RT-DETR study warned that "on a dataset this small, a single training run's score should not
be reported without either a repeated run or its epoch-variation figure." GMO-DETR makes the
point much more forcefully than RT-DETR did:

- Three identical baseline runs: **0.489, 0.390, 0.133**.
- Two identical `flipud0+neg300+1280` runs: **0.528, 0.128**.
- 2 of 18 runs (11%) collapsed entirely, early-stopping at epochs 84 and 76.
- Within-run epoch jitter is normal (0.016–0.042) even in the collapsed runs, so **the epoch-
  variation check alone does not catch these failures** — only a repeated run does.

Wave 1 of this study, before repeats, appeared to show "+0.101 significant" for both `flipud0`
and `neg300`. Both shrank to +0.038 and +0.007 once a second seed was added. Had the study
stopped at one run per configuration, it would have reported two confident findings that do not
exist.

### Object queries: the RT-DETR result does not transfer

Reducing queries to 30 was RT-DETR's largest gain (+0.046, confirmed twice). On GMO-DETR it
scored 0.387 against a 0.440 baseline and *doubled* epoch jitter (0.042 vs 0.019). The
mechanism explains it: RT-DETR's gain came from re-fitting a **pretrained** 300-query decoder
whose capacity was mismatched to a 1.11-object dataset. A from-scratch decoder never acquires
that mismatch, so there is nothing to correct — and 30 queries give the untrained matcher fewer
chances to find a positive assignment early in training.

### Small objects: the P2 head raises the floor, not the mean

Solder Ball AP@0.5 across all valid runs:

| | n | mean | sd | range |
|---|:---:|:---:|:---:|---|
| With P2 head | 6 | 0.267 | 0.070 | 0.198 – 0.362 |
| Without P2 head | 10 | 0.253 | 0.127 | 0.105 – 0.447 |

The means are indistinguishable. What does change is consistency: **no P2 run scored below
0.198, while the two plain baselines scored 0.105 and 0.109.** The stride-4 map appears to make
Solder Ball reliably detectable rather than better-detected on average — but with six runs
against a sd of 0.07–0.13, this is suggestive, not established. `flipud0+neg300+P2` does carry
the best mAP@0.5:0.95 of any repeated configuration (0.275), consistent with tighter boxes.

### Combinations still do not stack

As with RT-DETR, stacking beneficial changes did not accumulate: `flipud0` alone (0.478) beat
`flipud0+neg300` (0.430), and adding P2 recovered only to 0.475. Given the seed spread, the
honest reading is that all of these are the same number.

---

## Part 4 — Best GMO-DETR

**`flipud0 + neg300 + P2` (mAP@0.5 0.475 ± 0.059 over 3 runs, mAP@0.5:0.95 0.275)** — chosen
over `flipud0` (0.478 over 2 runs) because it has the most repeats, the best mAP@0.5:0.95, and
the most consistent Solder Ball performance. On this data the two are statistically identical
and the choice is a judgment call, not a measurement.

Best single run of the study: `flipud0+neg300+1280`, **0.528** — but its sibling seed collapsed
to 0.128, so it is the least trustworthy number in the table, not the most.

---

## Part 5 — Final comparison

| Model | mAP@0.5 | mAP@0.5:0.95 | Main observation |
|---|:---:|:---:|---|
| YOLO11s, 5-fold CV | 0.761 ± 0.021 | 0.500 ± 0.020 | baseline |
| RT-DETR, 5-fold CV | 0.755 ± 0.034 | 0.494 ± 0.036 | baseline; indistinguishable from YOLO |
| **Optimized RT-DETR (30 queries)** | **0.786** | **0.549** | best on this dataset |
| GMO-DETR | 0.440 ± 0.069 | 0.252 | from scratch, 441 training images |
| **Optimized GMO-DETR** | **0.475 ± 0.059** | **0.275** | +0.035, within its own noise |

**1. Highest mAP@0.5** — optimized RT-DETR, 0.786.
**2. Highest mAP@0.5:0.95** — optimized RT-DETR, 0.549.
**3. Best overall** — optimized RT-DETR (30 queries). YOLO11s and stock RT-DETR follow, tied
with each other; GMO-DETR is last by a wide margin.
**4. GMO-DETR's own improvement** — 0.440 → 0.475/0.478, about +0.035. This is **half its own
seed spread** and is not established.
**5. Does optimized GMO beat optimized RT-DETR?** — No. It is **0.311 mAP@0.5 and 0.274
mAP@0.5:0.95 behind**, and no tested change moved it more than 0.04.
**6. Is the difference meaningful?** — The GMO-vs-RT-DETR gap is ~15× the noise floor and is
unambiguous. The GMO-internal improvements are **not** meaningful: all fall inside ±0.069.
**7. Is GMO-DETR worth keeping?** — **Not as a detector for this dataset at its current size.**
Optimized RT-DETR is 0.31 mAP ahead, trains in less than half the time, and does not collapse.
GMO-DETR is worth keeping only as a research line, on the specific condition below.

---

## Part 6 — Conclusion

> **Optimized RT-DETR (30 object queries) is the best model on this dataset, at 0.786 mAP@0.5
> and 0.549 mAP@0.5:0.95. It leads optimized GMO-DETR by 0.311 mAP@0.5 — a difference far too
> large to be explained by measurement noise.**

Between YOLO11s (0.761 ± 0.021) and stock RT-DETR (0.755 ± 0.034) the difference *is* too small
to claim either is better; that conclusion from the RT-DETR study is unchanged. But nothing
about the GMO-DETR comparison is marginal.

The result should not be read as "the GMO-DETR architecture is worse." It is a statement about
**data scale and initialization**: a 17.9 M-parameter transformer with four novel modules,
trained from random initialization on 441 images, cannot match a 32 M-parameter model that
starts from COCO. The published 98.27% mAP@0.5 was obtained on 6,384 images — 14× more data.

---

## Part 7 — Future directions

1. **Do not adopt GMO-DETR for production on this dataset.** Ship optimized RT-DETR
   (30 queries). Revisit only if the dataset grows past a few thousand annotated defects.

2. **The highest-value work is data, not architecture.** GMO-DETR's deficit is an
   initialization and sample-count problem. Annotating another 500–1,000 defect images would do
   more for every model here than any further tuning, and is the only route by which GMO-DETR
   could become competitive.

3. **If GMO-DETR is pursued, pretrain the backbone.** GMONet could be pretrained on ImageNet or
   on the ~1,200 unlabeled defect-free boards via self-supervision (MAE/DINO), removing the one
   difference that plausibly accounts for the whole gap. This is the single experiment that
   would make the architecture comparison fair, and it is not cheap.

4. **Fix the training instability before any further GMO experiments.** An 11% collapse rate
   makes every measurement expensive. Gradient clipping, a longer warmup than 3 epochs, or a
   lower initial LR are the obvious first attempts; the collapses show normal epoch jitter, so
   they are silent failures that only a repeated run detects.

5. **Report GMO-DETR results as a mean of at least three runs.** Two is not enough at
   ±0.069 — this study needed three seeds to detect that its own wave-1 findings were noise.

6. **Carry the stride-4 head into RT-DETR instead.** It is recommendation #4 of the RT-DETR
   study and remains untested there. GMO-DETR's evidence for it is weak but not negative
   (Solder Ball never fell below 0.198 with it), and RT-DETR's far lower variance would
   actually resolve whether it works.

---

## Appendix — Reproducing

```
experiments/
  gmo_arch.py      GMO-DETR architecture, lifted verbatim from gmo_detr_full_pipeline.py,
                   plus Ultralytics registration and a parameterised model-YAML builder
  exp_gmodetr.py   one configurable run; every knob is an environment variable
  collect_gmo.py   aggregates runs/experiments/gmo_*/summary.json into the tables above
  run_gmo.sbatch   SLURM template (run_gmo_a40.sbatch for the a40 partition)
```

```bash
EXP_NAME=gmo_base sbatch experiments/run_gmo.sbatch
sbatch -J gmo_comb_p2 --export=ALL,EXP_NAME=gmo_comb_p2,FLIPUD=0,NEG_TOTAL=300,P2=1 \
       experiments/run_gmo.sbatch
```

Knobs: `NEG_TOTAL` `FLIPUD` `NUM_QUERIES` `P2` `IMGSZ` `BATCH` `EPOCHS` `SEED` `FOLD`.
Run directories: `gmo_base`(+`_s1`,`_s2`), `gmo_nq30`, `gmo_res1280`, `gmo_flipud0`(+`_s1`),
`gmo_neg300`(+`_s1`), `gmo_p2`(+`_s1`), `gmo_fu0_p2`, `gmo_comb`, `gmo_comb_p2`(+`_s1`,`_s2`),
`gmo_comb_1280`(+`_s1`). All 18 runs' curves and confusion matrices are under
`runs/experiments/`.
