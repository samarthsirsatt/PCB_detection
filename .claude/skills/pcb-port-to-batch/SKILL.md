---
name: pcb-port-to-batch
description: Convert the PCB defect-detection Kaggle/Colab notebook exports (8classes_kfold.py, rtdetr.py, pretrain_soldef_yolo.py, gmo_detr_full_pipeline.py) into scripts that run unattended under sbatch on a Linux HPC node. Use when repointing /kaggle/input or /kaggle/working paths, removing "!pip install" or "!git clone" shell magics, fixing device=[0,1] multi-GPU arguments, fixing relative-path writes such as gmo_detr_4a.yaml, replacing auto-discovery globs over /kaggle/input, splitting a script that runs several trainings back-to-back into separate jobs, or making a script find pretrained weights offline.
---

# Porting the PCB scripts to batch

Read `../../PRAJNA_CONFIG.md` §9 (script inventory) first — it has the exact line numbers for
every issue below, kept current as scripts change.

## When NOT to use this skill

- Setting up the venv/Spack/offline weights the ported script will need → `prajna-env`.
- Writing the `.sbatch` file that runs the ported script → `prajna-jobs`.
- Reading the finished run's metrics → `pcb-experiments`.

## Porting philosophy

**Keep the port mechanical.** These are notebook exports, not hand-written CLI tools — do not
add argparse, a `main()` guard, or restructure them, unless the user explicitly asks for more
than a working port (matches `AGENTS.md`'s existing convention). The smallest diff that runs
unattended under `sbatch` is the right diff. Resist the urge to "clean up" a script while
porting it — that's a separate task with its own review.

## The six transformations

Apply whichever of these actually appear in the script you're porting — check
`PRAJNA_CONFIG.md` §9 for which apply where.

1. **Paths.** Replace the Kaggle constants (`INPUT_ROOT`, `WORK`, `YOUR_COCO_JSON`,
   `YOUR_IMG_ROOT`, `NEG_IMG_ROOT`, and `gmo_detr_full_pipeline.py`'s `COCO_JSONS`/
   `IMAGE_ROOTS`/`NEG_IMAGE_DIR`/`OUT`) with the equivalents from `assets/hpc_env.py`
   (`COCO_JSON`, `IMG_ROOT`, `NEG_ROOT`, `DATA_ROOT`, `WORK`). Copy `hpc_env.py` into the repo
   alongside the script being ported and `from hpc_env import *` (or import the specific names)
   at the top, replacing the old constant block.

2. **Shell magics.** Delete every `!pip install ...` and `!git clone ...` line — they don't
   execute as `!`-prefixed shell escapes outside a notebook kernel anyway, and would fail
   loudly if they did. The dependency they represent (`ultralytics`, `einops`, `grad-cam`,
   `YOLO-26-CAM`) should already be staged by `prajna-env` — record it in `PRAJNA_CONFIG.md` §6
   if it isn't yet. `pretrain_soldef_yolo.py`'s
   `sys.path.insert(0, "/kaggle/working/YOLO-26-CAM")` becomes
   `sys.path.insert(0, str(PROJECT_ROOT / "offline" / "repos" / "YOLO-26-CAM"))`.

3. **Device selection.** Replace every hardcoded `device=[0,1]` (or any explicit device list)
   with `device=devices()` from `hpc_env.py`, which derives the right value from what SLURM
   actually granted (`CUDA_VISIBLE_DEVICES`) rather than assuming a fixed 2-GPU Kaggle
   notebook. See `prajna-jobs/references/partitions-qos.md` for why single-GPU is the right
   default at this project's scale.

4. **Relative-path writes.** `gmo_detr_full_pipeline.py` writes its model YAML with
   `open("gmo_detr_4a.yaml", "w")` (then loads it via `RTDETR("gmo_detr_4a.yaml")`) — a bare
   relative path that resolves against whatever CWD happens to be, which changes under
   `sbatch`. Make both the write and the read use an absolute path anchored at the run
   directory, e.g. `WORK / "gmo_detr_4a.yaml"`. Apply the same scrutiny to `project=` /
   `name=` arguments in any `.train()` call — they should resolve under `RUNS_ROOT`
   (`hpc_env.py`), not a bare relative string.

5. **Auto-discovery globs.** `pretrain_soldef_yolo.py`'s SolDef_AI discovery
   (`labeled_dirs[0]` after globbing `/kaggle/input`, via its `auto_find` helper) raises
   `IndexError` off-Kaggle — replace with an explicit path read from `PRAJNA_CONFIG.md` §7,
   guarded by `hpc_env.require(...)` so the failure is a clear message, not a stack trace deep
   in list indexing. `gmo_detr_full_pipeline.py`'s `/kaggle/input`-tree-printing cells are pure
   diagnostics with no downstream effect — delete them outright rather than porting them.

6. **Job decomposition.** A script that runs several trainings end-to-end
   (`8classes_kfold.py`: baseline + resolution sweep + 5-fold CV; `pretrain_soldef_yolo.py`:
   pretrain + 4 finetune arms) should not become one multi-hour `sbatch` job that loses
   everything if it dies at hour 10. See `references/job-decomposition.md` for the recommended
   split per script — this is a recommendation, not a rule; a single job is fine for an initial
   end-to-end smoke test.

## The smoke-test habit

Before submitting a freshly ported script at full scale, run it with a drastically reduced
epoch count (`EPOCHS=2` or similar — check whether the script already reads this from an env
var; if not, temporarily hand-edit it for the smoke test only) and `--time=00:20:00`. This
catches import errors, path mistakes, and device misconfiguration in minutes instead of hours.
Confirm a `results.csv` lands in `RUNS_ROOT` and the log shows no network calls before scaling
up.

## Idempotent dataset builds

All four scripts currently rebuild their YOLO-format dataset tree unconditionally on every run.
Given `TmpDisk=0` (no node-local scratch — every rebuild is Lustre I/O across ~1,750 small
files), make the build step skip when the target already exists:
```python
if not (OUT / "data.yaml").exists() or FORCE_REBUILD:
    build_dataset(...)
```
This matters most for the array-job CV split, where 5 concurrent tasks would otherwise each
rebuild the same tree redundantly.

## Escape hatch

If a script has an issue not covered by the six transformations above, port it the same
mechanical way — smallest fix that makes it run unattended — and add the new issue to
`PRAJNA_CONFIG.md` §9 so it's documented for next time.
