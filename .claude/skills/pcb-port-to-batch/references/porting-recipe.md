# Porting recipe — worked example

Walks through porting `rtdetr.py` end to end, since it's the simplest of the four (single
training call, no relative-path landmine, no auto-discovery glob). Apply the same pattern to
the others per `script-inventory.md`'s per-script notes.

## Before

```python
# rtdetr.py, original
INPUT_ROOT = Path("/kaggle/input")
WORK       = Path("/kaggle/working")
YOUR_COCO_JSON = "/kaggle/input/datasets/vidishagar/pcb-large-coco/annotations/instances_default.json"
YOUR_IMG_ROOT  = "/kaggle/input/datasets/vidishagar/pcb-large-dataset/categories"
NEG_IMG_ROOT   = "/kaggle/input/datasets/vidishagar/non-def-1200/OK PHOTOS_MERGED"
MODEL       = "rtdetr-l.pt"
IMG_SIZE    = 640
BATCH       = 8
EPOCHS      = 150
...
# !pip install ultralytics        <- shell magic, L177
...
model = RTDETR(MODEL)
model.train(
    data=..., epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, patience=30,
    device=[0,1],
    ...
    lr0=0.0001,
)
```

## After

```python
# rtdetr.py, ported
import sys
sys.path.insert(0, ".")   # or wherever hpc_env.py was copied alongside this script
from hpc_env import COCO_JSON, IMG_ROOT, NEG_ROOT, WORK, weights, devices

YOUR_COCO_JSON = str(COCO_JSON)
YOUR_IMG_ROOT  = str(IMG_ROOT)
NEG_IMG_ROOT   = str(NEG_ROOT)
MODEL       = weights("rtdetr-l.pt")
IMG_SIZE    = 640
BATCH       = 8
EPOCHS      = int(os.environ.get("PCB_EPOCHS", 150))   # smoke-test override, see SKILL.md
...
# (pip install line deleted — ultralytics is already staged in the venv)
...
model = RTDETR(MODEL)
model.train(
    data=..., epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, patience=30,
    device=devices(),
    ...
    lr0=0.0001,
)
```

Everything downstream of the constants (the dataset-building code, the augmentation config,
the actual training call's other arguments) is untouched — that's the "smallest diff that
runs" principle from `SKILL.md`.

## What changed, mapped to the six transformations

1. **Paths** — `YOUR_COCO_JSON`/`YOUR_IMG_ROOT`/`NEG_IMG_ROOT` now derive from `hpc_env`
   instead of being Kaggle literals.
2. **Shell magic** — `!pip install ultralytics` deleted; the dependency is staged by
   `prajna-env` ahead of time.
3. **Device** — `device=[0,1]` → `device=devices()`.
4. **Relative writes** — not present in `rtdetr.py`; see `gmo_detr_full_pipeline.py`'s entry in
   `script-inventory.md` for a case where it is.
5. **Auto-discovery globs** — not present in `rtdetr.py`; see `pretrain_soldef_yolo.py`.
6. **Job decomposition** — not needed; `rtdetr.py` is already a single training run.

Also added: `PCB_EPOCHS` env-var override, so the smoke-test habit (`SKILL.md`) doesn't
require hand-editing the file each time — set `PCB_EPOCHS=2` for a smoke test, leave unset for
the real 150-epoch run. Only add this override where it doesn't already exist; don't introduce
it speculatively into scripts that don't need repeated smoke-testing.

## Verifying the port

1. `python -c "import ast; ast.parse(open('rtdetr.py').read())"` — confirms it's still valid
   Python after edits.
2. `grep -n "/kaggle/" rtdetr.py` — should return nothing.
3. `grep -n "^!" rtdetr.py` — should return nothing (no shell magics left).
4. Smoke test per `SKILL.md`: `PCB_EPOCHS=2`, submit via `prajna-jobs/templates/gpu_job.sbatch`
   with `--time=00:20:00`, confirm a `results.csv` appears under `RUNS_ROOT` and the job's
   `.err` shows no network timeouts.
