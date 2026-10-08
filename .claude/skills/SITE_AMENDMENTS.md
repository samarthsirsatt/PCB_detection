# SITE_AMENDMENTS — where the skillset disagrees with the actual cluster

Append-only record of deviations found between what the skills / `PRAJNA_CONFIG.md` /
`Prajna_AIML_HPC_User_Manual.md` **claim** about this cluster and what was **observed** on it.

**Precedence.** `PRAJNA_CONFIG.md` still wins over any skill's `SKILL.md` or reference doc.
This file wins over both, but only for the specific facts listed here, and only until someone
folds an entry back into the config and deletes it from this file. Everything not listed here
is unaffected — read the config as written.

**Why this file exists instead of edits to the skills:** the deviations below were found in a
single verification pass before any setup had been done, by an agent that had not yet earned
the right to rewrite four skills' worth of assumptions. Recording beats rewriting until a real
training run confirms which corrections actually hold.

**How to use an entry.** Each has a `VERDICT:` line — `BLOCKER` (the documented path fails
outright), `WRONG` (documented value is factually incorrect), or `UNNECESSARY` (documented work
solves a problem this cluster doesn't have). Do what `Action:` says. If an entry turns out to be
wrong or gets fixed upstream, add a new dated entry saying so — never edit a past one.

---

## 2026-08-26 — first landing on Prajna, pre-setup verification

Observed on `login2.prajna.iitb.ac.in` as `samarth.sirsat` / group `agipml`. Nothing had been
installed yet; no venv, no `offline/`, no `runs/`. All commands below are read-only and were
re-runnable at the time of writing.

### A1 — Spack has no Python, CUDA, or torch  ·  VERDICT: BLOCKER

Amends: `PRAJNA_CONFIG.md` §4, §5 · `prajna-env/references/spack-and-python.md` ·
`prajna-jobs/templates/gpu_job.sbatch` L26 (and the same line in the other two templates)

The templates and config assume `spack load python@3.11`. **There is no `python` package in
Spack on this cluster**, nor `py-torch`, `cuda`, or `cudnn`:

```
$ source /lustre-flash/apps/spack/share/spack/setup-env.sh
$ spack find python      # ==> Error: No package matches the query: python
$ spack find py-torch    # ==> Error: No package matches the query: py-torch
$ spack find cuda cudnn  # ==> Error: No package matches the query: cuda cudnn
```

Spack 1.0.1 has only bootstrap-level packages installed (gcc, perl, autoconf, tar, xz, …) under
`linux-rocky8-nehalem / no compilers`.

**Action:** use `spack load anaconda3` instead — the one usable interpreter on the system:

```
$ spack load anaconda3   # anaconda3@2023.09-0
$ python3 -V             # Python 3.11.5
```

It ships numpy 1.24.3, scikit-learn 1.3.0, pandas 2.0.3, PIL 9.4.0, yaml 6.0, tqdm 4.65.0.
It does **not** ship torch, ultralytics, cv2, or pycocotools — those still need installing into
the project venv.

Do **not** fall back to system `python3`: it is **3.6.8**, far too old for current Ultralytics.

Related: config §4's `[FROM-MANUAL]` compiler list (`gcc@8.5.0/12.4.0/13.3.0/14.2.0`,
`nvhpc@23.11/24.11`, `oneapi@2024.2.1/2025.0.1`) does not match `spack compilers`, which
registers exactly one: `gcc@15.1.0`.

### A2 — the cluster has working internet egress  ·  VERDICT: UNNECESSARY

Amends: `PRAJNA_CONFIG.md` §6 · `prajna-env/references/offline-staging.md` ·
`prajna-env/scripts/verify_offline.sh`

The offline-staging apparatus (wheels, vendored repos, `--no-index --find-links`) solves a
problem this cluster does not have. From the login node:

```
pypi.org                       HTTP/2 200
files.pythonhosted.org         HTTP/2 200
github.com                     HTTP/2 200
objects.githubusercontent.com  HTTP/2 404   <- bare HEAD to the root; NOT a block
```

**Action:** a plain `pip install ultralytics` into the venv is expected to work. Fill config §6's
egress block with the above rather than treating it as `[TODO(day-one)]`.

**Caveat — not yet disproved:** egress was only tested *from the login node*. Compute nodes may
be firewalled differently. Until a GPU job has actually downloaded something, keep staging
`yolo11s.pt` / `yolo11n.pt` / `rtdetr-l.pt` into `offline/weights/` as cheap insurance — the
config §6 note about `yolo11n.pt` being needed by Ultralytics' AMP allclose check on *every*
`.train()` call (including GMO-DETR's) still stands regardless of egress.

### A3 — partition table is wrong in several rows  ·  VERDICT: WRONG

Amends: `PRAJNA_CONFIG.md` §3 · `prajna-jobs/references/partitions-qos.md`

Actual `sinfo -o "%20P %5a %10l %10G %D %N"`:

| partition | qos | nodes | GPUs/node | TIMELIMIT | vs. config §3 |
|---|---|---|---|---|---|
| `dgx` | `dgx` | 9 | 8 | 6-00:00:00 | resolves the `[TODO]` "9 nodes A100" row — the name is `dgx` |
| `dgx-mpi` | `dgx-mpi` | 3 | 8 | 6-00:00:00 | **not in config at all** |
| `a40` | `a40` | **19** | 4 | 4-00:00:00 | config says 20 nodes |
| `l40*` | `l40` | **6** | 8 | **2-00:00:00** | config says 7 nodes, and `MaxTime=UNLIMITED` `[FROM-MANUAL]` |
| `interactive` | `interactive` | 3 | 8 / 4 | 4:00:00 | **not in config at all** |
| `debug` | `debug` | 1 | 4 | 30:00 | **not in config at all** |

`l40` is still the default partition (`l40*`) — config §3 is right about that.

The `[TODO]` row for "10 nodes L4 24GB" has **no matching partition** in `sinfo`, even though an
`l4` QOS exists (see A4). Treat L4 as unavailable until proven otherwise.

**Action:** prefer `l40` for anything interactive-feeling — a `--test-only` probe placed an l40
job **immediately**, vs ~5 h queue on `a40` and ~3 days on `dgx`. Observed queue at the time:
49 running on a40, 44 on dgx, 16 on l40.

### A4 — QOS invariant CONFIRMED, but the failure mode is different  ·  VERDICT: WRONG (detail)

Amends: `PRAJNA_CONFIG.md` §3 · `prajna-jobs/references/triage.md`

The invariant itself is **correct and now confirmed** — every partition is locked to its
same-named QOS:

```
$ for p in dgx dgx-mpi a40 l40 interactive debug; do scontrol show partition $p | grep -o 'AllowQos=[^ ]*'; done
AllowQos=dgx  AllowQos=dgx-mpi  AllowQos=a40  AllowQos=l40  AllowQos=interactive  AllowQos=debug
```

This user holds all of them: `a40, debug, dgx, dgx-mpi, gh200, interactive, l4, l40, normal`
(`MaxJobs=12`, `MaxSubmit=20`). Note `gh200` and `l4` QOS exist with no corresponding partition.

**What the docs get wrong is the symptom.** Config §3 and the manual describe a mismatch as a job
that "sits PENDING forever" with
`Reason=Job's_QOS_not_permitted_to_use_this_partition`. Observed behaviour is an **immediate
rejection at submit time** — the job never enters the queue:

```
$ sbatch --test-only --partition=l40 --qos=normal ...
allocation failure: Invalid qos specification
```

**Action:** when triaging, a QOS mismatch will show up as a failed `sbatch`, not a stuck job. If
a job *is* stuck PENDING, the cause is something else — don't burn time on the QOS line.

### A5 — `--time` over the partition cap is not rejected by `--test-only`  ·  VERDICT: WRONG (trap)

Amends: `prajna-jobs/references/slurm-cheatsheet.md`

`--time=3-00:00:00` on `l40` (2-day cap) was accepted by `--test-only`, which reported a normal
start time rather than an error. Whether a real submission rejects it or the job is silently
killed at the cap was **not** tested.

**Action:** set `--time` under the partition cap deliberately; do not rely on SLURM to catch it.
Caps: `l40` 2 d · `a40` 4 d · `dgx`/`dgx-mpi` 6 d · `interactive` 4 h · `debug` 30 min.

### A6 — OS is Rocky 8.10, not Rocky 9  ·  VERDICT: WRONG (cosmetic)

Amends: `prajna-env/SKILL.md` frontmatter description ("Rocky 9, Spack, SLURM")

`Rocky Linux 8.10 (Green Obsidian)`, kernel `4.18.0-553.el8_10.x86_64`. Matters only if someone
picks a wheel or container by distro version. Left unedited — the skill's `description:` field is
what the skill loader matches on, and rewriting it risks the skill not triggering.

### A7 — `debug` partition rejects on a group submit limit  ·  VERDICT: noted, not diagnosed

```
$ sbatch --test-only --partition=debug --qos=debug --gres=gpu:1 --time=00:20:00 ...
sbatch: error: AssocGrpSubmitJobsLimit
allocation failure: Job violates accounting/QOS policy
```

A **group**-level cap (`agipml`), not this user's own — the same probe succeeded on l40/a40/dgx.
May clear on its own. Don't plan a workflow around `debug` being available.

### A8 — `PROJECT_ROOT` is not `~/pcb`  ·  VERDICT: WRONG (site fact)

Amends: `PRAJNA_CONFIG.md` §1

The payload landed at **`/home/agipml/samarth.sirsat/DDP/pcb`**, not the `~/pcb` that config §1
gives as the default and that `pcb-port-to-batch/assets/hpc_env.py` falls back to:

```python
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.home() / "pcb"))
```

**No code change is needed** — `hpc_env.py` reads the env var first, and resolves every path
correctly once it is set. Verified: `COCO_JSON`, `IMG_ROOT`, `NEG_ROOT` all resolve; `weights()`
falls back to bare names when `offline/weights/` is absent; `devices()` returns the right value
for `cpu` / single-GPU / multi-GPU / SLURM-granted `CUDA_VISIBLE_DEVICES`.

**Action:** always export before running anything, including in every `.sbatch`:

```
export PROJECT_ROOT=/home/agipml/samarth.sirsat/DDP/pcb
export SCRATCH_ROOT=/scratch/samarth.sirsat/pcb    # does not exist yet — mkdir on day one
```

Filesystem facts at first landing: `/home` used 10.14 G, **no quota enforced** (`quota`/`limit`
both `0k`) — config §2's "quota'd" assumption did not hold. `/scratch` is 250 G total, 212 G
available, and `/scratch/samarth.sirsat` **does not exist yet**.

### A9 — three dataset filenames arrived with `&` stripped  ·  VERDICT: FIXED, watch for recurrence

Amends: nothing — transfer artifact, recorded so the next sync doesn't silently repeat it.

The rsync from the dev machine deleted the literal `&` from three filenames under
`data/pcb_all_work/categories/`, breaking their match against `instances_default.json`:

```
RED & GROUND wire no solder.jpg    -> RED GROUND wire no solder.jpg
j2 DRY SOLDER & SOLDER SHORT.jpg   -> j2 DRY SOLDER SOLDER SHORT.jpg
Q6 &R8 Solder Boll.jpg             -> Q6 R8 Solder Boll.jpg
```

Renamed back on 2026-08-26; JSON and disk now map 1:1 (551/551, zero orphans either way).

**Why this was dangerous:** `8classes_kfold.py` L121 guards resolution with
`assert found >= len(images)*0.98`. At 548/551 = 99.46 % the assert **passes**, so three images
and five annotations would have been dropped **silently** — 2× Component No Solder, 1× Component
Solder Dry, 1× Solder Short, 1× Solder Ball, all in classes the 8-class study can least spare.

**Action:** after any future rsync, re-check before trusting the data:

```
python3 - <<'EOF'
import json, os
d = json.load(open("data/pcb_all_work/coco/annotations/instances_default.json"))
disk = {os.path.join(r, f) for r, _, fs in os.walk("data/pcb_all_work/categories") for f in fs}
js = {im["file_name"] for im in d["images"]}
print("missing:", sorted(js - disk)); print("orphan:", sorted(disk - js))
EOF
```

The transfer chain, not rsync itself, is the suspect — `&` is shell-significant. Worth fixing at
the source before the next sync.

---

## 2026-08-26 (later) — found while doing day-one setup

### A10 — there is no writable scratch tier  ·  VERDICT: BLOCKER (worked around)

Amends: `PRAJNA_CONFIG.md` §1 (`SCRATCH_ROOT`), §2, §8 · `Prajna_AIML_HPC_User_Manual.md` L189-192

The manual says "two directories are available (i.e. /home and /scratch)" and config §1 sets
`SCRATCH_ROOT=/scratch/<user>/pcb`. **Neither is usable by this user:**

```
$ mkdir -p /scratch/samarth.sirsat/pcb          # Permission denied
$ mkdir -p /lustre-scratch/samarth.sirsat/pcb   # Permission denied
```

`/scratch` is **not** the user scratch tier at all — it is a root-owned NFS mount holding SLURM
admin state (`slurmctld.log`, `slurm_conf/`, `sacct_jobs.csv`). The real Lustre scratch is
**`/lustre-scratch`** (820 T, 812 T free), which is also root-owned; it contains exactly one
per-user directory (`compiling-ganesh`, uid 1004), so per-user dirs there appear to be
**admin-provisioned on request** rather than self-served.

**Action / workaround in use:** point `SCRATCH_ROOT` at `$PROJECT_ROOT/scratch` on `/home`.
This is safe here *because* `/home` turned out to have **no enforced quota** (A8) and 580 T free
— but it inverts config §8's placement rule, which sends regenerable YOLO trees to scratch
specifically to keep them off `/home`. Two consequences:

- The 3-month reaper does **not** apply to anything we write, so nothing is auto-cleaned.
  Built YOLO trees under `$PROJECT_ROOT/scratch/` must be deleted by hand when done.
- `hpc_env.py` already defaults `SCRATCH_ROOT` to `PROJECT_ROOT/scratch` when the env var is
  unset, so it needs no change — but the `.sbatch` templates use `${SCRATCH_ROOT:?}` and will
  abort unless it is exported.

**Ask an admin for a `/lustre-scratch/$USER` directory** — that is the correct long-term fix and
would restore config §8 as written.

### A11 — a working venv already existed, at a different path  ·  VERDICT: WRONG (path)

Amends: `PRAJNA_CONFIG.md` §5 (`VENV=$PROJECT_ROOT/env/pcb`)

The venv is at **`~/envs/pcb`**, not `$PROJECT_ROOT/env/pcb`. It was built 2026-08-25, before
this verification pass, and is **fully provisioned** — no install work was needed:

```
torch 2.6.0+cu124 · torchvision 0.21.0+cu124 · ultralytics 8.4.127 · numpy 2.4.6
opencv-python 5.0.0.93 · scipy 1.17.1 · matplotlib 3.11.1 · pandas 3.0.5 · PyYAML 6.0.3
einops 0.8.2 (GMO-DETR) · grad-cam 1.5.7 + yolo_cam (pretrain_soldef, pip-installed —
so its `!git clone YOLO-26-CAM` step is already satisfied and must not be re-added)
```

Its `pyvenv.cfg` shows it was created from the Spack **anaconda3** interpreter — independently
confirming A1's recommendation. Every module the four scripts import is present.
`pycocotools` is absent and **not needed**: all four parse COCO with plain `json`.

**Action:** use `PCB_VENV="${PCB_VENV:-$HOME/envs/pcb}"`; the `.sbatch` templates were updated
to source that instead of `$PROJECT_ROOT/env/pcb`. Update config §5's `VENV=` line.

### A12 — first `torch`/`torchvision` import off Lustre is very slow  ·  VERDICT: noted

A cold `import torch; import torchvision` on the **login node** exceeded **2 minutes** and was
killed; `pip list` (metadata only) returns instantly. Expected for many-small-file reads over
Lustre with a cold cache.

**Action:** don't mistake a slow first import for a hang, and don't set a tight `--time` on a
job whose first act is importing torch. If it proves slow on compute nodes too, that is an
argument for asking an admin for `/lustre-scratch` space (A10) and staging the venv there.

---

## 2026-08-26 (later still) — first GPU job. **A2 was wrong.**

### A13 — CORRECTS A2: compute nodes have NO egress; offline staging is MANDATORY  ·  VERDICT: BLOCKER

Amends: **A2 above (do not act on it)** · `PRAJNA_CONFIG.md` §6

A2 concluded from a login-node probe that egress works and offline staging was "unnecessary".
**That conclusion was wrong for the only place it matters.** Measured inside a real `l40` GPU job
(`sbatch`, job 295266):

```
https://pypi.org/simple/                UNREACHABLE
https://github.com/                     UNREACHABLE
https://objects.githubusercontent.com/  UNREACHABLE
```

The **login node has egress; compute nodes do not.** This is exactly the split-reachability trap
config §6 warned about, and it is why `prajna-env/SKILL.md`'s "stage regardless of what the probe
says — *Deviate when:* nothing" rule exists. That rule was right and A2 should not have
second-guessed it.

**Practical consequence:** any bare weight name reaching a compute node is a hard failure —
Ultralytics cannot download `rtdetr-l.pt`, `yolo11s.pt`, `yolo11n.pt` (needed by the AMP allclose
check on *every* `.train()` call, all four scripts), or `Arial.ttf`.

**Action — done 2026-08-26,** staged from the login node into `$PROJECT_ROOT/offline/weights/`:

```
rtdetr-l.pt  64M  sha256=6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f
yolo11s.pt   19M  sha256=85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5
yolo11n.pt  5.4M  sha256=0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
Arial.ttf   756K  (also copied to offline/ultralytics_cfg/ — Ultralytics looks in YOLO_CONFIG_DIR)
```

Source: `https://github.com/ultralytics/assets/releases/download/v8.3.0/<name>`. Every `.sbatch`
must `ln -sf $PROJECT_ROOT/offline/weights/*.pt .` into its run dir so bare-name lookups resolve,
and set `YOLO_CONFIG_DIR` to a **writable** directory (see A14).

Do **not** `pip install` from inside a job. Install on the login node, into the venv, only.

### A14 — `YOLO_CONFIG_DIR` must exist and be writable, or Ultralytics silently uses `/tmp`

Observed in job 295266:

```
WARNING ⚠️ user config directory '.../scratch/ultralytics_cfg/Ultralytics' is not writable,
using '/tmp/Ultralytics'.
```

The path did not exist. The fallback to node-local `/tmp` is silent-ish and per-node, so
`Arial.ttf` staged elsewhere would not be found. **Action:** `mkdir -p "$YOLO_CONFIG_DIR"` in the
job before Python starts, and keep `Arial.ttf` inside it.

### A15 — GPU/driver facts, now confirmed on hardware  ·  fills `PRAJNA_CONFIG.md` §3/§5 TODOs

From job 295266 on `cn40-l40`:

```
GPU            NVIDIA L40S, 47.7 GB VRAM, compute capability (8, 9)
driver         570.86.15,  CUDA 12.8
torch          2.6.0+cu124  ->  is_available=True, device_count=1
sanity         4096^2 matmul on device: OK (7.4 s incl. cold CUDA context)
devices()      returns 0 under --gres=gpu:1   (hpc_env.py behaves correctly on real hardware)
```

torch's cu124 build against a CUDA 12.8 driver is fine (minor-version compatibility). The
existing venv needs no rebuild.

**One contradiction with config §2:** it states compute nodes have `TmpDisk=0` and therefore no
node-local scratch. The node reported `TMPDIR=/tmp` with **20 GB free** on its root filesystem.
`TmpDisk=0` is a SLURM *accounting* value, not proof `/tmp` is absent. Do not build a staging
strategy on this — `/tmp` is node-local, unmanaged, and may be wiped between jobs — but it does
explain why the Ultralytics `/tmp` fallback in A14 succeeded rather than crashing.

### A16 — the AMP-check asset is `yolo26n.pt`, not `yolo11n.pt`  ·  VERDICT: WRONG

Amends: `PRAJNA_CONFIG.md` §6 · `prajna-env/SKILL.md` INVARIANTS

Config §6 and `prajna-env/SKILL.md` both state — as an INVARIANT — that every script needs
`yolo11n.pt` for Ultralytics' AMP allclose check. **Not true for ultralytics 8.4.127**, the
version in the venv. Observed in job 295269:

```
WARNING ⚠️ AMP: checks skipped. Offline and unable to download YOLO26n for AMP checks.
Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False.
```

It wants **`yolo26n.pt`**. Two further corrections to the invariant as written:

1. A missing AMP asset **does not fail the run** — Ultralytics skips the check and proceeds with
   `amp=True`. The invariant's "Missing it fails all four" is too strong for this version.
2. `yolo26n.pt` is **not** in the `v8.3.0` assets release (404). It is in **`v8.4.0`**:
   `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`

**Action — done 2026-08-26:** `yolo26n.pt` (5.3 M) staged alongside the others. Keep `yolo11n.pt`
too — harmless, and correct for older ultralytics.

**Watch this:** the warning ties a skipped AMP check to possible zero-mAP/NaN losses. The smoke
run *did* produce near-zero mAP, but that is fully explained by training for 2 epochs. If a
**full-length** run also lands at zero mAP, retry with `amp=False` before assuming a data or
model bug — at roughly 2x VRAM in fp32, which the L40S's 47.7 GB can absorb at batch 8.

### A17 — `workers=9` exceeds the 8 CPUs granted  ·  VERDICT: minor

```
UserWarning: This DataLoader will create 9 worker processes ... suggested max ... is 8
```

Ultralytics defaults `workers=8` per the args dump but instantiates 9 processes against
`--cpus-per-task=8`. Harmless at this scale (the job completed in 4 min) — if dataloader stalls
ever appear, either raise `--cpus-per-task` or pass `workers=6`.

---

## 2026-08-26 — SETUP IS FUNCTIONAL: first end-to-end GPU training run succeeded

Job **295269**, `l40`/`cn40-l40`, `rtdetr.py` ported, `PCB_EPOCHS=2` smoke — **COMPLETED, exit 0,
4 min 03 s**, no network access required at any point.

What this proves end-to-end on real hardware: dataset build on the compute node (identical
407 train / 101 val to the login-node dry run) → staged `rtdetr-l.pt` resolved via
`hpc_env.weights()` with **no download attempted** → `devices()` returned `0` → RT-DETR trained
on the L40S → `results.csv`, `best.pt`, curves and confusion matrices landed in
`$PROJECT_ROOT/runs/rtdetr8_640/` per config §8.

Training was **learning**, which is the part that matters at 2 epochs:

```
epoch  train/cls_loss  val/cls_loss  recall
1        19.02          15.26        0.018
2         1.17           3.42        0.179
```

mAP@0.5 = 0.0004 is meaningless this early and is **not** a defect — compare only after a
full-length run (baseline to beat: RT-DETR 0.770 / 0.556 per `pcb-experiments`).

**Remaining known gaps:** `pretrain_soldef_yolo.py` still blocked (no SolDef_AI dataset);
`8classes_kfold.py` and `gmo_detr_full_pipeline.py` not yet ported; no `/lustre-scratch` quota
(A10), so `$PROJECT_ROOT/scratch/` must be cleaned by hand.

---

## Verified working at first landing — do not re-litigate these

Recorded so a later session doesn't re-derive them. All confirmed 2026-08-26, pre-setup.

- **Dataset is complete and intact.** 551 images across 12 class subfolders (per-class counts
  match `AGENTS.md` and config §7 exactly), 1,195 negatives, 445 M total. Zero empty files; all
  1,746 JPEGs carry valid SOI+EOI markers.
- **COCO JSON is sound.** Parses; 551 images / 604 annotations / 12 categories; no orphan
  annotations, no unknown category ids. Four images legitimately carry zero annotations
  (`U2 Sensor pin not solder (2)`, `U3 display lead no solder`, `C5 solder short (2)`,
  `U1 IC solder short (3)`) — present in the source annotations, not a transfer fault.
- **The 8-class data-prep pipeline runs clean.** `8classes_kfold.py`'s COCO→filter→stratified
  split→YOLO-label-write logic was executed verbatim against the real data (scratchpad output,
  no training): 508 images / 565 annotations → 407 train / 101 val, **all 8 classes populated in
  both splits**, 0 out-of-range normalized bboxes, 0 orphan images, negatives split 96/24.
  The dataset build is not where a first training run will fail.
- **All skill scripts and templates are syntactically valid** — `collect_runs.py`, `probe.sh`,
  `verify_offline.sh`, and all three `.sbatch` templates.
- **The four training scripts still fail `py_compile`** on `!pip install` magics
  (L179/L177/L208/L425). Expected — that is the un-ported Kaggle state `pcb-port-to-batch`
  exists to fix, not transfer damage.
