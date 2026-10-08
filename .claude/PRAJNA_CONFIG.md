# PRAJNA_CONFIG — site + project facts for the PCB defect-detection work

Read this file before doing anything else. It holds every site- and project-specific value the
skills reference. **If this file disagrees with a skill's SKILL.md or a reference doc, this
file wins** — it reflects what was actually observed on the cluster.

**Except where `skills/SITE_AMENDMENTS.md` says otherwise.** That file records deviations found
on the cluster that have not yet been folded back into this one; it overrides this file for the
specific facts it lists (as of 2026-08-26: §1 PROJECT_ROOT, §2 quota, §3 partitions, §4/§5 Spack
has no Python, §6 egress works). Read it before acting on §1–§6.

Status legend used throughout:
- `[CONFIRMED yyyy-mm-dd]` — verified by running a command on the cluster
- `[FROM-MANUAL]` — stated in `Prajna_AIML_HPC_User_Manual.md`, not independently verified
- `[TODO(day-one)]` — unknown; fill in via the `prajna-env` skill's discovery probe

When you resolve a `[TODO(day-one)]`, replace it in place and stamp it
`[CONFIRMED yyyy-mm-dd]` with the date you ran the command. Do not delete the command that
produced the value — keep it as a comment so the next person can re-verify.

---

## 1. Identity and roots

```
GROUP=          [TODO(day-one)]   # from: id -gn
USER=           [TODO(day-one)]   # from: whoami
HOME_ROOT=/home/<group>/<user>                      [TODO(day-one)]
PROJECT_ROOT=/home/<group>/<user>/pcb               [TODO(day-one)]
SCRATCH_ROOT=/scratch/<user>/pcb                     [TODO(day-one)]
```

No passwords, tokens, or key material belong in this file, ever. See §11.

---

## 2. Filesystem policy

- `/home` is quota'd. Measured quota: `[TODO(day-one)]` — run `lfs quota -h -u $USER /home` or `quota -s`.
- `/scratch`: anything not accessed in the last three months is **permanently deleted**
  `[FROM-MANUAL]` (manual, "Points to remember").
- Compute nodes have `TmpDisk=0` `[FROM-MANUAL, manual L638, scontrol show node cn20-a40]` —
  there is **no node-local scratch**. The usual "stage dataset to `$TMPDIR`" trick does not
  exist here; all dataset I/O during training crosses the Lustre filesystem.
- Placement rule for what goes where: see §8.

---

## 3. Partitions / QOS / GRES

**INVARIANT:** `--qos` must equal `--partition`. Confirmed in the manual's own
`scontrol show job` example (manual, "Monitoring jobs" section):
```
Reason=Job's_QOS_not_permitted_to_use_this_partition_(l40_allows_l40_not_norm al)
```
A partition/QOS mismatch produces exactly this reason string and the job sits PENDING forever.

| partition | qos | nodes | GPUs/node | GPU model | VRAM | status |
|---|---|---|---|---|---|---|
| `l40` | `l40` | 7 | 8 | L40S | 48GB | `[CONFIRMED-manual]` DEFAULT partition (`Default=YES`, manual L653) |
| `a40` | `a40` | 20 | 4 | A40 | 48GB | `[CONFIRMED-manual]` |
| ? | ? | 10 | 2 | L4 | 24GB | `[TODO(day-one)]` partition name not in manual |
| ? | ? | 9 | 8 | A100 | 40/80GB | `[TODO(day-one)]` DGX A100 partition name not in manual |

`DefaultTime=NONE`, `MaxTime=UNLIMITED` on `l40` (manual L654-655) — in practice this means
**`--time` is mandatory**; an unset or too-generous time limit only hurts you under backfill
scheduling (longer expected runtime = later start for everyone, including you).

Discover the unconfirmed rows and your real per-user limits with (all read-only):
```
sinfo -o "%20P %5a %10l %10G %D %N"
scontrol show partition <partition> | grep -E "AllowQos|MaxTime|DefaultTime"
sacctmgr show assoc user=$USER format=partition,qos,maxjobs,maxwall,maxsubmit
```

Queue policy / walltime caps published externally:
https://hpcverse.iitb.ac.in/queue-policy/prajna — `[TODO(day-one)]` read and record anything
that adds constraints beyond what's above.

---

## 4. Spack

```
source /lustre-flash/apps/spack/share/spack/setup-env.sh
```

Compilers available `[FROM-MANUAL]`: `gcc@8.5.0/12.4.0/13.3.0/14.2.0`, `nvhpc@23.11/24.11`,
`oneapi@2024.2.1/2025.0.1`.

Relevant installed packages `[TODO(day-one)]` — record verbatim output of:
```
spack find python
spack find py-torch
spack find cuda cudnn
```

---

## 5. Python environment

```
VENV=$PROJECT_ROOT/env/pcb                          [TODO(day-one)]
```

- Base interpreter: `[TODO(day-one)]` — record which (`spack load python@X.Y` vs system
  `python3`) and why, per `references/spack-and-python.md` in the `prajna-env` skill.
- torch version + CUDA build actually installed: `[TODO(day-one)]`.
- Install provenance: `online pip` | `--no-index --find-links offline/wheels` — record which.

---

## 6. Offline staging inventory

```
$PROJECT_ROOT/offline/
  weights/   yolo11s.pt  yolo11n.pt  rtdetr-l.pt      (sha256 recorded below)
  wheels/    ultralytics, einops, grad-cam + transitive deps
  repos/     YOLO-26-CAM/
  fonts/     Arial.ttf
```

Weight checksums `[TODO(day-one)]` (`sha256sum offline/weights/*.pt`):
```
yolo11s.pt   sha256=TODO
yolo11n.pt   sha256=TODO
rtdetr-l.pt  sha256=TODO
```

Egress probe results `[TODO(day-one)]` — see `prajna-env/references/discovery-probe.md`:
```
pypi.org                          = TODO
files.pythonhosted.org            = TODO
github.com                        = TODO
objects.githubusercontent.com     = TODO   (this is where Ultralytics release assets live —
                                             reachable github.com with blocked objects.* is a
                                             real and confusing state; test both separately)
api.anthropic.com                 = yes (implied — Claude Code is running here)
```

Why `yolo11n.pt` is staged even though nothing trains it: Ultralytics' AMP allclose check
instantiates `YOLO("yolo11n.pt")` at the start of every `.train()` call, including the
from-scratch GMO-DETR run. Missing it fails all four scripts, not just the ones that use a
pretrained YOLO backbone.

---

## 7. Dataset

```
$PROJECT_ROOT/data/pcb_all_work/
  coco/annotations/instances_default.json
  categories/              551 images, 12 class subfolders
  OK PHOTOS_MERGED/        1,195 negatives
```

12-class counts: Component Crack 6 · Component Damage 11 · Component Liftup 22 ·
Component Missing 43 · Component No Solder 68 · Component Solder Dry 133 · LED Damage 7 ·
Polarity Wrong 36 · RYB Wrong Sequence 24 · Solder Ball 34 · Solder Short 152 · Tombstone 15.

The 8-class study (`8classes_kfold.py`, `rtdetr.py`) drops the four rarest (≤15 instances):
Component Crack, Component Damage, LED Damage, Tombstone.

SolDef_AI `Labeled/` dataset: **NOT PRESENT** anywhere in this workspace or on the cluster.
`pretrain_soldef_yolo.py` cannot run until it is sourced separately (it is the public SolDef_AI
dataset, not part of `pcb_all_work`). `[TODO]` — update when/if it's transferred.

---

## 8. Output placement rule

| what | where | why |
|---|---|---|
| built YOLO trees, per-fold copies, `OUT=.../dataset`, mpl/torch caches, per-job CWD | `$SCRATCH_ROOT` | regenerable from `data/` in minutes; large; exactly what the 3-month reaper is for |
| `runs/` — `best.pt`, `last.pt`, `results.csv`, `args.yaml`, plots, confusion matrices | `$PROJECT_ROOT/runs/` | irreplaceable (hours of GPU time each), small (tens of MB); the reaper must never touch these |
| SLURM `%x.%j.out` / `.err` | `$PROJECT_ROOT/logs/` | kilobytes; needed for triage weeks later |
| the dataset itself | `$PROJECT_ROOT/data/` | manually transferred and painful to redo; never let a reaper near it |

Do not write `runs/` to `/scratch` and sync back — write it straight to `/home` so a killed job
still leaves `last.pt` durable. Full rationale: `prajna-jobs/references/run-layout.md` (via
`pcb-experiments/references/run-layout.md`).

---

## 9. Script inventory

Source of truth for the `pcb-port-to-batch` skill. All four scripts live in
`PCB_defect_detection_internship/` and are Colab/Kaggle notebook exports — flat, top-to-bottom,
no `argparse`, no `def main`, no `if __name__ == "__main__"`.

| script | path vars @ lines | network deps | GPU arg | trainings | notes |
|---|---|---|---|---|---|
| `8classes_kfold.py` | L15-20 (`INPUT_ROOT`,`WORK`,`YOUR_COCO_JSON`,`YOUR_IMG_ROOT`,`NEG_IMG_ROOT`); `MODEL` L27 | `yolo11s.pt`; `!pip install ultralytics` L179, L272 | `device=[0,1]` L187, L215, L311 | 1 base (150ep) + 2 res-sweep (150ep @960 b12, @1280 b6) + 5-fold CV (120ep ×5) | recommended split: 3 jobs (baseline, res-sweep, CV array) |
| `rtdetr.py` | same block, L15-20 (`NEG_IMG_ROOT` differs), `MODEL="rtdetr-l.pt"` L27 | `rtdetr-l.pt`; `!pip install ultralytics` L177 | `device=[0,1]` L185 | 1 (150ep, batch 8) | single job |
| `pretrain_soldef_yolo.py` | SolDef root **auto-discovered** by globbing `/kaggle/input` (L118-119, `auto_find` helper L219) — will `IndexError` off-Kaggle | `yolo11s.pt`; `!pip install ultralytics` L208; **`!git clone https://github.com/rigvedrs/YOLO-26-CAM.git`** L335; `!pip install grad-cam` L338; `sys.path.insert(0, "/kaggle/working/YOLO-26-CAM")` L337 | none (Ultralytics default) | 5 trainings (100 + 4×150 epochs) | **BLOCKED** — SolDef_AI `Labeled/` not present anywhere; recommended split: pretrain → finetune A/B (`-d afterok:<id>`) → EigenCAM viz (no GPU needed) |
| `gmo_detr_full_pipeline.py` | config block at L679-688 (`COCO_JSONS`, `IMAGE_ROOTS`, `NEG_IMAGE_DIR`, `OUT="/kaggle/working/dataset"`); more `/kaggle/input` globbing at L639, L644-646, L666-667, L671, L674; `project="/kaggle/working/runs"` L917 | `!pip install -q ultralytics einops` L425 | `device = 0 if torch.cuda.is_available() else "cpu"` L904 — already single-GPU-aware | 1 (200ep, batch 8) | **writes model YAML to a relative path**: `open("gmo_detr_4a.yaml","w")` L532, then `RTDETR("gmo_detr_4a.yaml")` L540 — breaks the instant CWD differs under `sbatch`; also `MAX_NEGATIVES=120` set twice (L798, L881) |

Full transformation checklist and the `hpc_env.py` shim: `pcb-port-to-batch/SKILL.md` and its
`references/`.

---

## 10. Observed cluster behaviour (append-only log)

All four skills should append here when something surprising happens — an error not in
`triage.md`, a partition/QOS combination that behaved unexpectedly, a quota or reaper surprise,
a package that Spack provides differently than expected. Never edit past entries; add new ones
with a date.

```
- yyyy-mm-dd  <what broke> | <exact error string> | <what fixed it>
```

- 2026-08-26  first landing, pre-setup verification | 9 deviations from this config and the
              manual (Spack has no python; egress works; partition table wrong; QOS mismatch
              rejects at submit rather than PENDING; PROJECT_ROOT is not ~/pcb; 3 dataset
              filenames lost their "&" in transfer) | recorded in full, with evidence commands
              and actions, in `skills/SITE_AMENDMENTS.md` — not duplicated here
- 2026-10-03  job allocated on cn22-a40 never started | `srun: error: Task launch for StepId=…
              failed on node cn22-a40: Job credential expired` | node-side fault (likely clock
              skew); resubmit with `--exclude=cn22-a40`
- 2026-10-03  `interactive` partition job sat PENDING 30+ min | `(ReqNodeNotAvail,
              UnavailableNodes:cn11-dgx,cn41-l40)` | cancelled; short GPU jobs go to
              `--partition=a40 --qos=a40` instead, which started within seconds

---

## 11. Secrets policy

This file holds usernames, group names, absolute paths, partition/QOS names, and package
versions. **Nothing else, ever.**

Never write into this file (or any skill file, or a pasted report):
- the contents of `~/.ssh/*`, `~/.netrc`, `~/.config/gcloud`, `/tmp/krb5cc_*`, or
  `~/.kaggle/kaggle.json`
- a wholesale `env`/`printenv` dump — `ANTHROPIC_API_KEY` is in that environment by
  construction since Claude Code runs on the login node
- any token value — if a script ever needs one (HF, W&B), record the **environment variable
  name** here, never the value, and source it from a `chmod 600` file outside this repo

No SSH/scp/rsync logic exists anywhere in this skillset — all transfer to/from the cluster is
done manually by the user and is out of scope.
