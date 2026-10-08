# Offline staging

Stage everything into `$PROJECT_ROOT/offline/` regardless of what the egress probe found —
reachability at setup time isn't reachability when an unattended job runs at 3am.

```
$PROJECT_ROOT/offline/
  weights/            yolo11s.pt  yolo11n.pt  rtdetr-l.pt
  wheels/             ultralytics, einops, grad-cam + transitive deps
  repos/              YOLO-26-CAM/
  fonts/              Arial.ttf
  requirements.lock   pip freeze output
```

## Weights — three files, not two

Stage `yolo11s.pt`, `rtdetr-l.pt`, **and `yolo11n.pt`**. The third one is easy to miss: nothing
in the four scripts references it directly, but Ultralytics' AMP allclose check instantiates
`YOLO("yolo11n.pt")` at the start of every single `.train()` call — including the from-scratch
GMO-DETR run, which needs no pretrained weights of its own for anything else. Skip staging it
and all four scripts fail identically at the same point.

Getting the files onto the cluster:
- If egress works: `wget`/`curl` them directly from Ultralytics' GitHub releases
  (`objects.githubusercontent.com`) into `offline/weights/`.
- If egress is blocked: download them on a machine that has internet (they're small, tens of
  MB each) and transfer them up manually along with everything else — this is out of scope for
  these skills, but note it as a transfer-list item.

Record `sha256sum offline/weights/*.pt` output in `PRAJNA_CONFIG.md` §6 once staged.

Also stage `Arial.ttf` into wherever `YOLO_CONFIG_DIR` points — Ultralytics fetches this
silently for plot annotation whenever a script calls `.train(..., plots=True)` (which
`gmo_detr_full_pipeline.py` does).

## Resolution order, three mechanisms deep

1. **Primary — absolute path.** The `hpc_env.py` shim's `weights(name)` function
   (`pcb-port-to-batch/assets/hpc_env.py`) returns the absolute path to a staged file if it
   exists, falling back to the bare name so the script still works unmodified if you ever run
   it somewhere with real internet. This handles every explicit `YOLO(MODEL)` /
   `RTDETR(MODEL)` call.
2. **Backstop — CWD symlink.** The AMP allclose check calls `YOLO("yolo11n.pt")` with no way to
   inject a path. `prajna-jobs/templates/gpu_job.sbatch` `cd`s into a per-job run directory and
   symlinks `offline/weights/*.pt` into it before launching Python, so Ultralytics' bare-name
   resolution finds it in the current directory.
3. **Last resort — `amp=False`.** Only if both of the above are somehow unavailable. State the
   cost honestly when you reach for this: disabling AMP switches training to fp32, roughly
   doubling VRAM usage and slowing training meaningfully. This is fallback #3, never the first
   thing you reach for.

## Wheels

From a machine with internet (this cannot be done from an offline cluster, and building CUDA
wheels for `torch` from the Windows workspace isn't practical — see the caveat below):

```bash
pip download --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all: \
  -d offline/wheels ultralytics einops grad-cam
```

**Caveat, flag on day one rather than at install time:** `torch`'s CUDA wheels are ~2.5GB and
tied to a specific CUDA build (`+cu121`, `+cu124`, etc.). If the cluster genuinely has no
egress, torch has to come from Spack (`spack find py-torch` — see `spack-and-python.md`) or
from a wheel fetched elsewhere and transferred up manually; it is not something to try
downloading through this workspace's network. Decide which path you're on during the day-one
probe, not partway through an install.

Install offline with:
```bash
pip install --no-index --find-links "$PROJECT_ROOT/offline/wheels" ultralytics einops grad-cam
```

If egress does work, install normally, but still run the `pip download` above into
`offline/wheels/` as insurance, and still produce `offline/requirements.lock` via `pip freeze`
— the project has never had a reproducibility artifact before this.

## Vendored repo

`pretrain_soldef_yolo.py` clones `https://github.com/rigvedrs/YOLO-26-CAM.git` and inserts it
into `sys.path` for its EigenCAM visualization cells. Clone it once, anywhere with internet,
into `offline/repos/YOLO-26-CAM/`; the porting skill repoints the `sys.path.insert` there.

This only affects `pretrain_soldef_yolo.py`'s final visualization step, which the recommended
job decomposition (`pcb-port-to-batch/references/job-decomposition.md`) splits into its own
GPU-free job. If this repo is missing, the four actual trainings in that script still run —
only the visualization step is blocked.

## Env vars that prevent surprise fetches

Set these in every `.sbatch` preamble (already in `gpu_job.sbatch`):

```bash
export YOLO_CONFIG_DIR="$PROJECT_ROOT/offline/ultralytics_cfg"
export MPLCONFIGDIR="$SCRATCH_ROOT/mplcache"
export TORCH_HOME="$PROJECT_ROOT/offline/torch"
export HF_HUB_OFFLINE=1
```

These also keep Ultralytics' settings file and matplotlib's font cache off `$HOME` and off
Lustre's metadata server, which matters given `TmpDisk=0` pushes all such small-file traffic
onto the parallel filesystem.

## Verifying

Run `../scripts/verify_offline.sh` after staging — it checks the three weights exist and match
recorded checksums, `offline/wheels/` is non-empty, and the venv imports `ultralytics` with
`torch.cuda.is_available()` true (run this check under `srun`, not the login node).
