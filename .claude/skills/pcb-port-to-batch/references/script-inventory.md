# Script inventory

Canonical copy lives in `PRAJNA_CONFIG.md` §9 — update both if a script changes. This file adds
the fuller reasoning behind each entry; the config table is the quick-reference version.

All four scripts are in `PCB_defect_detection_internship/`, are Colab/Kaggle notebook exports,
and have no `argparse`, no `def main`, no `if __name__ == "__main__"` guard — they run flat,
top to bottom.

## `8classes_kfold.py` (343 lines)

- Path constants at L15-20: `INPUT_ROOT`, `WORK`, `YOUR_COCO_JSON`, `YOUR_IMG_ROOT`,
  `NEG_IMG_ROOT`. `MODEL = "yolo11s.pt"` at L27.
- `!pip install ultralytics` at L179 and L272 — delete both.
- Three training calls, all `device=[0,1]`: baseline (L182-192, EPOCHS=150), resolution sweep
  (L210-227/230, called at 960px batch 12 and 1280px batch 6), 5-fold CV (L308-311,
  CV_EPOCHS=120, K_FOLDS=5).
- Recommended decomposition: 3 jobs — baseline, resolution sweep, CV as a `--array=0-4` job
  (each fold ~2h instead of 5 sequential folds ~10h).

## `rtdetr.py` (204 lines)

- Same path-constant block as above (L15-20), `NEG_IMG_ROOT` differs (points at the 1,200
  negatives variant). `MODEL = "rtdetr-l.pt"` L27.
- `!pip install ultralytics` L177 — delete.
- Single training call L180-190, `device=[0,1]` L185, batch 8, 150 epochs, lr0=0.0001.
- Single job — no decomposition needed.

## `pretrain_soldef_yolo.py` (442 lines) — currently blocked

- Path constants L15-29, same block as `8classes_kfold.py` plus `PRETRAIN_EPOCHS=100`,
  `FINETUNE_EPOCHS=150`.
- **SolDef_AI root is not a constant** — it's auto-discovered by globbing `/kaggle/input` via
  an `auto_find` helper (L219) and indexed `labeled_dirs[0]` (L118-119). This raises
  `IndexError` immediately off-Kaggle once ported, unless replaced per transformation #5.
- **The SolDef_AI `Labeled/` dataset is not present anywhere in this workspace or on the
  cluster.** This script cannot actually run until that dataset is sourced separately (it's the
  public SolDef_AI dataset, unrelated to `pcb_all_work`). Porting the mechanics is still useful
  now — it just can't be smoke-tested until the data exists. Record its arrival in
  `PRAJNA_CONFIG.md` §7.
- `!pip install ultralytics` L208, **`!git clone https://github.com/rigvedrs/YOLO-26-CAM.git`**
  L335, `!pip install grad-cam` L338 — all deleted; the clone becomes a pre-staged directory
  (`prajna-env/references/offline-staging.md`) and `sys.path.insert(0, "/kaggle/working/YOLO-26-CAM")`
  at L337 repoints there.
- Five trainings back to back: pretrain (L211-215) then four finetune arms (L311-320,
  L417-427) — `finetune_A_baseline`, `finetune_B_pretrained`, and two more without negatives.
  No `device=` argument anywhere (uses Ultralytics' default single-GPU behavior already).
- Recommended decomposition: pretrain job → finetune A/B job (`-d afterok:<pretrain_id>`) →
  EigenCAM visualization job (no GPU required, and doesn't need YOLO-26-CAM's absence to block
  the trainings above it).

## `gmo_detr_full_pipeline.py` (923 lines)

- Defines the GMO-DETR architecture from scratch (GhostConv, DMambaOut/GatedCNN, CAFF+CAA,
  GSConv, TAIFI/TSSA) and registers it into Ultralytics' `RTDETR` via a patched
  `parse_model`. Config block is not at the top — it's at L679-688: `COCO_JSONS`,
  `IMAGE_ROOTS`, `NEG_IMAGE_DIR`, `OUT="/kaggle/working/dataset"`. More `/kaggle/input`
  references scattered at L639, L644-646, L666-667, L671, L674 — these are diagnostic
  tree-printing cells with no downstream effect and can be deleted rather than ported.
- `!pip install -q ultralytics einops` L425 — delete.
- **No pretrained checkpoint** — model is built fresh from a YAML written at runtime:
  `open("gmo_detr_4a.yaml", "w")` L532-533, then `RTDETR("gmo_detr_4a.yaml")` L540. Both use a
  bare relative path — the single most important fix in this file, since it silently breaks
  the moment `sbatch`'s CWD differs from wherever this was written (see triage.md's entry for
  `FileNotFoundError: gmo_detr_4a.yaml`).
- Single training call L904-921: `device = 0 if torch.cuda.is_available() else "cpu"` (L904) —
  already single-GPU-aware, no fix needed here, unlike the other three scripts.
  `project="/kaggle/working/runs"` L917 needs the same absolute-path treatment as the YAML.
  200 epochs, batch 8, imgsz 640, AdamW, lr 1e-4.
- `MAX_NEGATIVES=120` set twice (L798 and L881) — the second silently wins; worth flagging to
  whoever edits this next, though not itself a porting blocker.
- Single job — no decomposition needed; it's already one coherent training run.

## Network-dependency summary

| script | needs from network |
|---|---|
| `8classes_kfold.py` | `yolo11s.pt` (auto-download unless staged), 2× pip install |
| `rtdetr.py` | `rtdetr-l.pt`, pip install |
| `pretrain_soldef_yolo.py` | `yolo11s.pt`, git clone, 2× pip install |
| `gmo_detr_full_pipeline.py` | pip install only (no pretrained weights of its own) |

All four additionally need `yolo11n.pt` staged, invisibly, for Ultralytics' AMP allclose check
at the start of every `.train()` call — see `prajna-env/references/offline-staging.md`.
